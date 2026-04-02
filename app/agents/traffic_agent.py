"""트래픽 라우터 — 인텐트 분류 후 LangGraph / LangChain 분기."""
from __future__ import annotations

from opik import track

_INTENT_PROMPT = """사용자 메시지를 보고 인텐트를 분류하세요.

- dog_symptom : 강아지(반려견)의 증상, 질환, 건강, 응급 상황 관련 질문
- general     : 그 외 모든 질문 (인사, 잡담, 강아지 무관 질문 등)

메시지: {message}

인텐트를 dog_symptom 또는 general 중 하나만 출력하세요 (다른 텍스트 없이):"""

_GENERAL_SYSTEM_PROMPT = """당신은 친절한 AI 어시스턴트입니다.
사용자의 질문에 간결하고 정확하게 답변하세요."""


@track(name="classify_intent")
def classify_intent(message: str) -> str:
    """사용자 메시지의 인텐트를 분류합니다.

    Returns:
        "dog_symptom" | "general"
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from app.core.config import settings
    from pydantic import SecretStr

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
        temperature=0,
    )
    chain = ChatPromptTemplate.from_template(_INTENT_PROMPT) | llm
    result = chain.invoke({"message": message})
    intent = result.content.strip().lower()

    if "dog_symptom" in intent:
        return "dog_symptom"
    return "general"


@track(name="general_chain")
def run_general_chain(message: str) -> str:
    """일반 질문에 LangChain으로 직접 응답합니다."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from app.core.config import settings
    from pydantic import SecretStr

    llm = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
    )
    response = llm.invoke([
        SystemMessage(content=_GENERAL_SYSTEM_PROMPT),
        HumanMessage(content=message),
    ])
    return response.content
