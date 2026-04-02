"""StateGraph 기반 강아지 증상 분석 파이프라인.

흐름:
  collect_symptoms
    ├─ (simple)  → search_rag → generate_report → END
    ├─ (complex) → parallel_analysis → generate_report → END
    └─ (follow)  → ask_follow_up → END  (사용자 응답 대기)
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.prompts import DOG_SYMPTOM_SYSTEM_PROMPT
from app.agents.search_agent import search_symptoms
from app.agents.tools import get_pet_breed_info


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class PetProfile(TypedDict, total=False):
    breed: Optional[str]
    age: Optional[str]
    weight: Optional[str]


class SymptomState(TypedDict):
    messages: Annotated[list, add_messages]
    question_count: int       # 에이전트가 추가 질문한 횟수
    pet_profile: PetProfile   # 반려견 프로필 (품종, 나이, 체중)
    is_complex: bool          # 복잡한 케이스 여부 (병렬 분석 트리거)


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOpenAI:
    from app.core.config import settings
    from pydantic import SecretStr
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
    )


def _has_enough_info(state: SymptomState) -> bool:
    """HumanMessage가 2개 이상이면 충분한 정보로 판단한다."""
    human_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    return human_count >= 2


_COMPLEX_KEYWORDS = [
    "경련", "발작", "혈변", "혈뇨", "호흡곤란", "의식", "심한", "갑자기",
    "여러", "동시에", "먹지 않", "안 먹", "쓰러",
]


def _detect_complexity(state: SymptomState) -> bool:
    """복잡 증상 키워드가 있거나 breed가 있으면 병렬 분석 사용."""
    text = " ".join(m.content for m in state["messages"] if isinstance(m, HumanMessage))
    has_keyword = any(kw in text for kw in _COMPLEX_KEYWORDS)
    has_breed = bool(state.get("pet_profile", {}).get("breed"))
    return has_keyword or has_breed


def _extract_pet_profile(state: SymptomState) -> PetProfile:
    """메시지에서 품종/나이/체중 정보를 간단히 추출한다."""
    import re
    text = " ".join(m.content for m in state["messages"] if isinstance(m, HumanMessage))

    breed = None
    breed_patterns = [
        r"(말티즈|푸들|포메라니안|치와와|비숑|시바이누|골든리트리버|래브라도|진도개|보더콜리|닥스훈트|불독|프렌치불독|요크셔테리어|시츄|웰시코기)"
    ]
    for pattern in breed_patterns:
        m = re.search(pattern, text)
        if m:
            breed = m.group(1)
            break

    age = None
    age_match = re.search(r"(\d+)\s*살|(\d+)\s*개월", text)
    if age_match:
        age = age_match.group(0)

    return {"breed": breed, "age": age, "weight": None}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def collect_symptoms(state: SymptomState) -> dict:
    """첫 번째 노드: pet_profile과 복잡도를 감지한다."""
    profile = _extract_pet_profile(state)
    is_complex = _detect_complexity(state)
    return {
        "pet_profile": profile,
        "is_complex": is_complex,
    }


def ask_follow_up(state: SymptomState) -> dict:
    """LLM이 추가 질문을 생성한다."""
    follow_up_questions = [
        "언제부터 그런 증상이 있었나요?",
        "밥이나 물은 먹고 있나요?",
        "다른 이상한 행동도 보이나요? (기침, 설사, 무기력 등)",
    ]
    count = state["question_count"]
    if count < len(follow_up_questions):
        question = follow_up_questions[count]
    else:
        llm = _get_llm()
        resp = llm.invoke(
            [SystemMessage(content="강아지 증상 추가 정보를 얻기 위한 짧은 질문 하나만 하세요.")]
            + state["messages"]
        )
        question = resp.content

    return {
        "messages": [AIMessage(content=question)],
        "question_count": state["question_count"] + 1,
    }


def search_rag(state: SymptomState) -> dict:
    """BM25+Vector+RRF 하이브리드 검색으로 관련 질환 정보를 가져온다 (simple path)."""
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    search_result = search_symptoms.invoke({"search_query": user_content})
    return {
        "messages": [SystemMessage(content=f"[검색 결과]\n{search_result}")]
    }


def parallel_analysis(state: SymptomState) -> dict:
    """3개의 서브에이전트를 병렬 실행한다 (complex path)."""
    import asyncio
    from app.agents.parallel_agents import run_parallel_analysis

    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    breed = state.get("pet_profile", {}).get("breed")

    results = asyncio.run(run_parallel_analysis(user_content, breed=breed))

    parts: list[str] = []
    if results.get("symptom_search"):
        parts.append(f"[증상 검색 결과]\n{results['symptom_search']}")
    if results.get("breed_info"):
        parts.append(f"[품종 취약 질환]\n{results['breed_info']}")
    if results.get("urgency"):
        parts.append(f"[긴급도 사전 판단] {results['urgency']}")

    combined = "\n\n".join(parts) if parts else "관련 정보를 찾을 수 없습니다."
    return {
        "messages": [SystemMessage(content=f"[병렬 분석 결과]\n{combined}")]
    }


def generate_report(state: SymptomState) -> dict:
    """수집된 증상 + 검색 결과를 바탕으로 종합 리포트를 작성한다."""
    llm = _get_llm()
    report_instruction = SystemMessage(
        content=(
            DOG_SYMPTOM_SYSTEM_PROMPT
            + "\n\n위 대화와 검색 결과를 바탕으로 종합 리포트를 작성하세요. "
            "반드시 긴급도·의심 질환·가정 대처법·병원 방문 시 전달사항을 포함하세요."
        )
    )
    response = llm.invoke([report_instruction] + state["messages"])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# 라우팅
# ---------------------------------------------------------------------------


def route_after_collect(
    state: SymptomState,
) -> Literal["ask_follow_up", "search_rag", "parallel_analysis"]:
    """정보 충분도와 복잡도에 따라 다음 노드를 선택한다."""
    if not (state["question_count"] >= 3 or _has_enough_info(state)):
        return "ask_follow_up"
    if state.get("is_complex", False):
        return "parallel_analysis"
    return "search_rag"


# ---------------------------------------------------------------------------
# 그래프 구성
# ---------------------------------------------------------------------------


def build_symptom_graph(checkpointer=None):
    builder = StateGraph(SymptomState)

    builder.add_node("collect_symptoms", collect_symptoms)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("search_rag", search_rag)
    builder.add_node("parallel_analysis", parallel_analysis)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "collect_symptoms")
    builder.add_conditional_edges("collect_symptoms", route_after_collect)
    builder.add_edge("ask_follow_up", END)          # 사용자 응답 대기
    builder.add_edge("search_rag", "generate_report")
    builder.add_edge("parallel_analysis", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(checkpointer=checkpointer)
