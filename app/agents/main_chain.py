"""LangChain 메인 체인 — LangGraph를 서브모듈로 내장.

흐름:
  MainChain.astream()
      ├─ [새 thread] intent_chain (LangChain LCEL) → 인텐트 분류
      │
      ├─ intent == general
      │       └─ general_chain (LangChain LCEL) → 단발 LLM 응답
      │
      └─ intent == dog_symptom
              └─ symptom_graph (LangGraph Runnable) → 다단계 증상 분석
"""
from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.agents.symptom_pipeline import build_symptom_graph

# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------

_INTENT_PROMPT = """\
사용자 메시지를 보고 인텐트를 분류하세요.

- dog_symptom : 강아지(반려견)의 증상, 질환, 건강, 응급 상황 관련 질문
- general     : 그 외 모든 질문 (인사, 잡담, 강아지 무관 질문 등)

메시지: {message}

인텐트를 dog_symptom 또는 general 중 하나만 출력하세요 (다른 텍스트 없이):"""

_GENERAL_SYSTEM = "당신은 친절한 AI 어시스턴트입니다. 사용자의 질문에 간결하고 정확하게 답변하세요."


# ---------------------------------------------------------------------------
# MainChain
# ---------------------------------------------------------------------------


class MainChain:
    """LangChain LCEL 기반 메인 체인.

    - intent_chain  : ChatPromptTemplate | LLM | StrOutputParser  (LangChain)
    - general_chain : ChatPromptTemplate | LLM | StrOutputParser  (LangChain)
    - symptom_graph : build_symptom_graph()                       (LangGraph Runnable)
    """

    def __init__(self, llm: ChatOpenAI, checkpointer=None):
        # LangChain LCEL 체인 구성
        self.intent_chain = (
            ChatPromptTemplate.from_template(_INTENT_PROMPT)
            | llm
            | StrOutputParser()
        )

        self.general_chain = (
            ChatPromptTemplate.from_messages([
                ("system", _GENERAL_SYSTEM),
                ("human", "{message}"),
            ])
            | llm
            | StrOutputParser()
        )

        # LangGraph — LangChain Runnable 인터페이스 구현체이므로 서브모듈로 사용
        self.symptom_graph = build_symptom_graph(checkpointer=checkpointer)

    # ------------------------------------------------------------------
    # 인텐트 분류 (LangChain 체인)
    # ------------------------------------------------------------------

    async def _classify(self, message: str) -> str:
        """intent_chain을 비동기 실행하여 인텐트를 반환합니다."""
        result = await self.intent_chain.ainvoke({"message": message})
        return "dog_symptom" if "dog_symptom" in result.lower() else "general"

    # ------------------------------------------------------------------
    # 스트리밍 진입점
    # ------------------------------------------------------------------

    async def astream(
        self,
        message: str,
        thread_id: str,
        is_new_thread: bool,
    ) -> AsyncIterator[dict]:
        """메인 체인 스트리밍 실행.

        Yields:
            {"intent": str, "type": "general"|"langgraph", ...}
        """
        # 1) 인텐트 결정
        if is_new_thread:
            intent = await self._classify(message)
        else:
            intent = "dog_symptom"  # 기존 thread는 dog_symptom 유지

        # 2a) general → LangChain general_chain (단발 응답)
        if intent == "general":
            content = await self.general_chain.ainvoke({"message": message})
            yield {"intent": intent, "type": "general", "content": content}
            return

        # 2b) dog_symptom → LangGraph 서브모듈 (다단계 StateGraph)
        config = {"configurable": {"thread_id": thread_id}}
        input_state: dict = {"messages": [HumanMessage(content=message)]}
        if is_new_thread:
            input_state["question_count"] = 0

        async for chunk in self.symptom_graph.astream(
            input_state, config=config, stream_mode="updates"
        ):
            yield {"intent": intent, "type": "langgraph", "chunk": chunk}
