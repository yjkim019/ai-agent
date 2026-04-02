"""StateGraph 기반 강아지 증상 분석 파이프라인.

흐름:
  collect_symptoms → (정보 충분?) → search_rag → generate_report
                  ↘ (부족) → ask_follow_up → END (사용자 응답 대기)
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

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


class SymptomState(TypedDict):
    messages: Annotated[list, add_messages]
    question_count: int   # 에이전트가 추가 질문한 횟수


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


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def collect_symptoms(state: SymptomState) -> dict:
    """첫 번째 노드: 상태 초기화만 수행 (메시지는 이미 state에 있음)."""
    return {}


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
    """BM25+Vector+RRF 하이브리드 검색으로 관련 질환 정보를 가져온다."""
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    search_result = search_symptoms.invoke({"search_query": user_content})
    # 검색 결과를 SystemMessage로 삽입해 generate_report 노드에서 활용
    return {
        "messages": [SystemMessage(content=f"[검색 결과]\n{search_result}")]
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


def route_after_collect(state: SymptomState) -> Literal["ask_follow_up", "search_rag"]:
    """충분한 정보가 있거나 질문 횟수가 3회에 도달하면 검색으로 진행."""
    if state["question_count"] >= 3 or _has_enough_info(state):
        return "search_rag"
    return "ask_follow_up"


# ---------------------------------------------------------------------------
# 그래프 구성
# ---------------------------------------------------------------------------


def build_symptom_graph(checkpointer=None):
    builder = StateGraph(SymptomState)

    builder.add_node("collect_symptoms", collect_symptoms)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("search_rag", search_rag)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "collect_symptoms")
    builder.add_conditional_edges("collect_symptoms", route_after_collect)
    builder.add_edge("ask_follow_up", END)          # 사용자 응답 대기
    builder.add_edge("search_rag", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(checkpointer=checkpointer)
