import asyncio
import contextlib
from datetime import datetime
import json
import os
from typing import Optional
import uuid

from app.utils.logger import log_execution, custom_logger

from langgraph.errors import GraphRecursionError


def _configure_opik():
    """settings.OPIK 값을 기반으로 Opik 환경변수를 설정합니다."""
    from app.core.config import settings

    if settings.OPIK is None:
        return

    opik_settings = settings.OPIK
    if opik_settings.URL_OVERRIDE:
        os.environ["OPIK_URL_OVERRIDE"] = opik_settings.URL_OVERRIDE
    if opik_settings.API_KEY:
        os.environ["OPIK_API_KEY"] = opik_settings.API_KEY
    if opik_settings.WORKSPACE:
        os.environ["OPIK_WORKSPQCE"] = opik_settings.WORKSPACE
    if opik_settings.PROJECT:
        os.environ["OPIK_PROJECT_NAME"] = opik_settings.PROJECT


_configure_opik()


class AgentService:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        from app.core.config import settings
        from pydantic import SecretStr

        self.model = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
        )

        self.opik_tracer = None
        if settings.OPIK is not None:
            from opik.integrations.langchain import OpikTracer
            self.opik_tracer = OpikTracer(
                tags=["dog-symptom-agent"],
                metadata={"model": settings.OPENAI_MODEL},
            )

        self.checkpointer = None
        self.main_chain = None  # MainChain (LangChain + LangGraph 서브모듈)

    async def _init_checkpointer(self):
        """SQLite checkpointer 비동기 초기화 (첫 호출 시 한 번만 실행)"""
        if self.checkpointer is not None:
            return
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = await aiosqlite.connect("checkpoints.db")
        self.checkpointer = AsyncSqliteSaver(conn)

    def _init_main_chain(self):
        """MainChain 초기화 (checkpointer 준비 후 한 번만 실행)"""
        if self.main_chain is not None:
            return
        from app.agents.main_chain import MainChain
        assert self.checkpointer is not None, "checkpointer가 초기화되지 않았습니다."
        self.main_chain = MainChain(llm=self.model, checkpointer=self.checkpointer)

        if self.opik_tracer is not None:
            from opik.integrations.langchain import track_langgraph
            self.main_chain.symptom_graph = track_langgraph(
                self.main_chain.symptom_graph, self.opik_tracer
            )

    # ------------------------------------------------------------------
    # 실제 대화 로직
    # ------------------------------------------------------------------

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """MainChain을 통해 쿼리를 처리합니다.

        - MainChain이 인텐트를 분류하고 general/dog_symptom 경로를 선택합니다.
        - dog_symptom → LangGraph 서브모듈(StateGraph)로 위임합니다.
        """
        try:
            await self._init_checkpointer()
            self._init_main_chain()

            custom_logger.info(f"사용자 메시지: {user_messages}")

            # 기존 thread 여부 확인 (question_count 리셋 방지 + 인텐트 재분류 방지)
            config = {"configurable": {"thread_id": str(thread_id)}}
            existing_state = await self.checkpointer.aget(config)
            is_new_thread = existing_state is None

            async for event in self.main_chain.astream(
                message=user_messages,
                thread_id=str(thread_id),
                is_new_thread=is_new_thread,
            ):
                intent = event.get("intent", "")
                custom_logger.info(f"인텐트: {intent}, 타입: {event.get('type')}")

                if event["type"] == "general":
                    # LangChain general_chain 응답
                    yield self._done_event(content=event["content"])

                elif event["type"] == "langgraph":
                    # LangGraph 서브모듈 청크 파싱
                    chunk = event["chunk"]
                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for event_str in self._parse_chunk(chunk):
                            yield event_str
                    except Exception as e:
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        yield self._done_event(content="데이터 처리 중 오류가 발생했습니다.")

        except Exception as e:
            import traceback
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(traceback.format_exc())
            yield json.dumps({
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "metadata": {},
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e) if not isinstance(e, GraphRecursionError) else None,
            }, ensure_ascii=False)

    def _done_event(self, content: str, metadata: dict = None, message_id: str = None) -> str:
        return json.dumps({
            "step": "done",
            "message_id": message_id or str(uuid.uuid4()),
            "role": "assistant",
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }, ensure_ascii=False)

    def _parse_chunk(self, chunk: dict) -> list[str]:
        """LangGraph StateGraph 스트림 청크를 SSE 이벤트 문자열 리스트로 변환한다."""
        events: list[str] = []
        for step, event in chunk.items():
            if not event:
                continue
            messages = event.get("messages", []) if isinstance(event, dict) else []
            if not messages:
                continue
            message = messages[-1]

            if step == "model":
                tool_calls = getattr(message, "tool_calls", [])
                if tool_calls:
                    first_tool = tool_calls[0]
                    if first_tool.get("name") == "ChatResponse":
                        args = first_tool.get("args", {})
                        events.append(self._done_event(
                            content=args.get("content", ""),
                            metadata=self._handle_metadata(args.get("metadata")),
                            message_id=args.get("message_id"),
                        ))
                    else:
                        events.append(json.dumps({
                            "step": "model",
                            "tool_calls": [tc["name"] for tc in tool_calls],
                        }))

            elif step == "tools":
                events.append(
                    f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {message.content}}}'
                )

            elif step == "ask_follow_up":
                events.append(self._done_event(content=message.content))

            elif step == "generate_report":
                import json as _json
                try:
                    parsed = _json.loads(message.content)
                    content = parsed.get("content", message.content)
                    metadata = parsed.get("metadata", {})
                except (ValueError, AttributeError):
                    content = message.content
                    metadata = {}
                events.append(self._done_event(content=content, metadata=metadata))

        return events

    @log_execution
    def _handle_metadata(self, metadata) -> dict:
        custom_logger.info("========================================")
        custom_logger.info(metadata)
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result
