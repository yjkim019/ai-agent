from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from opik import track

from app.agents.symptom_pipeline import build_symptom_graph


@track(name="create_dog_agent")
def create_dog_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None):
    """
    강아지 증상 분석 StateGraph 에이전트를 생성합니다.

    Args:
        model: 초기화된 ChatOpenAI 인스턴스 (symptom_pipeline 내부에서 사용)
        checkpointer: 대화 이력 저장소

    Returns:
        컴파일된 LangGraph StateGraph
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    return build_symptom_graph(checkpointer=checkpointer)
