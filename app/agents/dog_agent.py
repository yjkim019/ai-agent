from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.agents.search_agent import search_symptoms
from app.agents.tools import get_pet_breed_info, find_nearby_vet
from app.agents.prompts import DOG_SYMPTOM_SYSTEM_PROMPT


@dataclass
class ChatResponse:
    """에이전트의 최종 응답 스키마."""
    message_id: str
    content: str
    metadata: dict[str, object]


def create_dog_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None):
    """강아지 증상 분석 에이전트를 생성합니다."""
    if checkpointer is None:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    return create_agent(
        model=model,
        tools=[search_symptoms, get_pet_breed_info, find_nearby_vet],
        system_prompt=DOG_SYMPTOM_SYSTEM_PROMPT,
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )
