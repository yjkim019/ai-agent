import asyncio
import contextlib
from datetime import datetime
import json
import os
import traceback
import uuid

from app.utils.logger import log_execution, custom_logger

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

# 도구 호출 총 횟수 제한 (ReAct 루프: 도구 1회 = LLM노드 + Tool노드 = 2 step)
_RECURSION_LIMIT = 15


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
        os.environ["OPIK_WORKSPACE"] = opik_settings.WORKSPACE
    if opik_settings.PROJECT:
        os.environ["OPIK_PROJECT_NAME"] = opik_settings.PROJECT


_configure_opik()


class AgentService:
    def __init__(self):
        from langchain_openai import ChatOpenAI
        from app.core.config import settings
        from pydantic import SecretStr

        # LLM 초기화
        self.model = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
        )

        # Opik 트레이서 초기화
        self.opik_tracer = None
        if settings.OPIK is not None:
            from opik.integrations.langchain import OpikTracer

            self.opik_tracer = OpikTracer(
                tags=["dog-symptom-agent"],
                metadata={"model": settings.OPENAI_MODEL}
            )

        # 대화 이력 저장소: process_query 첫 호출 시 async 초기화
        self.checkpointer = None
        self.agent = None
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    async def _init_checkpointer(self):
        """SQLite checkpointer 비동기 초기화 (첫 호출 시 한 번만 실행)"""
        if self.checkpointer is not None:
            return
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        conn = await aiosqlite.connect("checkpoints.db")
        self.checkpointer = AsyncSqliteSaver(conn)

    def _create_agent(self):
        """강아지 증상 분석 ReAct 에이전트 생성"""
        from app.agents.dog_agent import create_dog_agent
        assert self.checkpointer is not None, "checkpointer가 초기화되지 않았습니다. _init_checkpointer를 먼저 호출하세요."
        self.agent = create_dog_agent(
            model=self.model,
            checkpointer=self.checkpointer,
        )

        # Opik LangGraph 트래킹 적용
        if self.opik_tracer is not None:
            from opik.integrations.langchain import track_langgraph

            self.agent = track_langgraph(self.agent, self.opik_tracer)

    # -----------------------------------------------------------------------
    # SSE 이벤트 빌더
    # -----------------------------------------------------------------------

    @staticmethod
    def _done_event(content: str, metadata: dict | None = None,
                    message_id: str | None = None, error: str | None = None) -> str:
        """최종 응답(step=done) SSE JSON 문자열을 생성한다."""
        payload = {
            "step": "done",
            "message_id": message_id or str(uuid.uuid4()),
            "role": "assistant",
            "content": content,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }
        if error is not None:
            payload["error"] = error
        return json.dumps(payload, ensure_ascii=False)

    # -----------------------------------------------------------------------
    # 청크 파싱
    # -----------------------------------------------------------------------

    def _parse_chunk(self, chunk: dict):
        """에이전트 스트림 청크를 SSE 이벤트 문자열 리스트로 변환한다."""
        events: list[str] = []
        for step, event in chunk.items():
            if not event or step not in ("model", "tools"):
                continue
            messages = event.get("messages", [])
            if not messages:
                continue
            message = messages[0]

            if step == "model":
                tool_calls = message.tool_calls
                if not tool_calls:
                    continue
                first_tool = tool_calls[0]
                if first_tool.get("name") == "ChatResponse":
                    args = first_tool.get("args", {})
                    custom_logger.info(args)
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

        return events

    # -----------------------------------------------------------------------
    # 실제 대화 로직
    # -----------------------------------------------------------------------

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """LangChain Messages 형식의 쿼리를 처리하고 SSE 이벤트를 스트리밍한다."""
        # 초기화
        await self._init_checkpointer()
        self._create_agent()
        custom_logger.info(f"사용자 메시지: {user_messages}")

        config = {
            "configurable": {"thread_id": str(thread_id)},
            "recursion_limit": _RECURSION_LIMIT,
        }

        # 이전 세션에서 미완료된 tool_call이 있으면 사전 패치 (체크포인트 정합성 복구)
        await self._patch_pending_tool_calls(config)

        try:
            async for event in self._stream_agent(config, user_messages):
                yield event
        except GraphRecursionError:
            async for event in self._handle_recursion_fallback(config):
                yield event
        except Exception as e:
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(traceback.format_exc())
            await self._patch_pending_tool_calls(config)
            yield self._done_event("처리 중 오류가 발생했습니다. 다시 시도해주세요.", error=str(e))

    async def _stream_agent(self, config: dict, user_messages: str):
        """에이전트 스트림과 진행 큐를 동시에 처리하며 SSE 이벤트를 yield한다."""
        agent_iterator = self.agent.astream(
            {"messages": [HumanMessage(content=user_messages)]},
            config=config,
            stream_mode="updates",
        ).__aiter__()

        agent_task = asyncio.create_task(agent_iterator.__anext__())
        progress_task = asyncio.create_task(self.progress_queue.get())

        try:
            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                # --- progress 큐 이벤트 ---
                if progress_task in done:
                    try:
                        yield json.dumps(progress_task.result(), ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except (asyncio.CancelledError, Exception) as e:
                        if not isinstance(e, asyncio.CancelledError):
                            custom_logger.error(f"progress_task 오류: {e}")
                        progress_task = None

                # --- 에이전트 스트림 이벤트 ---
                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        break
                    except GraphRecursionError:
                        raise
                    except Exception as e:
                        custom_logger.error(f"agent_task 오류: {e}")
                        custom_logger.error(traceback.format_exc())
                        yield self._done_event("처리 중 오류가 발생했습니다. 다시 시도해주세요.", error=str(e))
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    for event in self._parse_chunk(chunk):
                        yield event
                    agent_task = asyncio.create_task(agent_iterator.__anext__())
        finally:
            # progress_task 정리
            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            # 큐에 남은 이벤트 드레인
            while not self.progress_queue.empty():
                try:
                    yield json.dumps(self.progress_queue.get_nowait(), ensure_ascii=False)
                except asyncio.QueueEmpty:
                    break

    async def _handle_recursion_fallback(self, config: dict):
        """GraphRecursionError 발생 시 미완료 tool_calls 패치 후 LLM 폴백 응답을 생성한다."""
        custom_logger.warning(
            f"GraphRecursionError: recursion_limit={_RECURSION_LIMIT} 도달. LLM 폴백 응답 생성."
        )
        try:
            state = await self.agent.aget_state(config)
            messages = list(state.values.get("messages", []))

            # 미완료 tool_calls에 대한 ToolMessage 보충 (체크포인트 정합성 복구)
            pending_tool_calls = self._find_pending_tool_calls(messages)
            if pending_tool_calls:
                custom_logger.info(f"미완료 tool_call {len(pending_tool_calls)}건 보충")
                patch_messages = [
                    ToolMessage(
                        content="[도구 호출 횟수 제한으로 실행되지 않았습니다]",
                        tool_call_id=tc["id"],
                    )
                    for tc in pending_tool_calls
                ]
                await self.agent.aupdate_state(
                    config, {"messages": patch_messages}, as_node="tools",
                )
                messages.extend(patch_messages)

            messages.append(SystemMessage(
                content="도구 호출 횟수 제한에 도달했습니다. "
                        "지금까지 수집된 정보를 바탕으로 사용자에게 최선의 답변을 생성하세요. "
                        "추가 도구 호출 없이 텍스트로만 답변하세요."
            ))
            response = await self.model.ainvoke(messages)
            yield self._done_event(response.content)

        except Exception as fallback_err:
            custom_logger.error(f"LLM 폴백 응답 생성 실패: {fallback_err}")
            yield self._done_event("검색 횟수 제한에 도달했습니다. 질문을 더 구체적으로 다시 시도해주세요.")

    # -----------------------------------------------------------------------
    # 체크포인트 정합성 유틸리티
    # -----------------------------------------------------------------------

    @staticmethod
    def _find_pending_tool_calls(messages: list) -> list[dict]:
        """메시지 목록에서 ToolMessage 응답이 없는 미완료 tool_calls를 찾는다."""
        answered_ids: set[str] = {
            msg.tool_call_id for msg in messages if isinstance(msg, ToolMessage)
        }
        pending: list[dict] = []
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    if tc["id"] not in answered_ids:
                        pending.append(tc)
        return pending

    async def _patch_pending_tool_calls(self, config: dict) -> None:
        """체크포인트에 미완료 tool_calls에 대한 더미 ToolMessage를 패치한다."""
        try:
            state = await self.agent.aget_state(config)
            messages = list(state.values.get("messages", []))
            pending = self._find_pending_tool_calls(messages)
            if not pending:
                return
            custom_logger.info(f"미완료 tool_call {len(pending)}건 체크포인트 패치")
            patch_messages = [
                ToolMessage(content="[오류로 인해 실행되지 않았습니다]", tool_call_id=tc["id"])
                for tc in pending
            ]
            await self.agent.aupdate_state(
                config, {"messages": patch_messages}, as_node="tools",
            )
        except Exception as patch_err:
            custom_logger.error(f"체크포인트 패치 실패: {patch_err}")

    # -----------------------------------------------------------------------
    # 기타 유틸리티
    # -----------------------------------------------------------------------

    @log_execution
    def _handle_metadata(self, metadata) -> dict:
        custom_logger.info(metadata)
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result
