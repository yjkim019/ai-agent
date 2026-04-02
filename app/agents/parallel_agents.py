"""3개의 서브에이전트를 asyncio.gather로 병렬 실행하는 모듈.

서브에이전트:
  - symptom_analyzer  : 증상 검색 (BM25+kNN+RRF)
  - breed_researcher  : 품종별 취약 질환 조회
  - urgency_judge     : 긴급도 판단 (LLM)
"""
from __future__ import annotations

import asyncio
from typing import Optional


# ---------------------------------------------------------------------------
# 서브에이전트 1: 증상 분석
# ---------------------------------------------------------------------------


async def symptom_analyzer(symptom_text: str) -> dict:
    """BM25+kNN+RRF 하이브리드 검색으로 관련 질환 정보를 반환합니다."""
    from app.agents.search_agent import search_symptoms

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, search_symptoms.invoke, {"search_query": symptom_text}
    )
    return {"type": "symptom_search", "result": result}


# ---------------------------------------------------------------------------
# 서브에이전트 2: 품종 연구
# ---------------------------------------------------------------------------


async def breed_researcher(breed: Optional[str]) -> dict:
    """품종명이 제공된 경우 취약 질환 정보를 반환합니다."""
    if not breed:
        return {"type": "breed_info", "result": None}

    from app.agents.tools import get_pet_breed_info

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None, get_pet_breed_info.invoke, {"breed_name": breed}
    )
    return {"type": "breed_info", "result": result}


# ---------------------------------------------------------------------------
# 서브에이전트 3: 긴급도 판단
# ---------------------------------------------------------------------------

_EMERGENCY_KEYWORDS = [
    "경련", "발작", "의식", "호흡곤란", "심한 출혈", "혈변", "혈뇨",
    "쓰러", "기절", "마비", "움직이지", "골절",
]

_URGENCY_PROMPT = """아래 증상 설명을 보고 긴급도를 판단하세요.
긴급도 등급만 JSON으로 반환하세요: {{"urgency": "high"|"medium"|"low"|"observe"}}

- high   : 즉시 병원 방문 필요 (경련, 의식 저하, 심한 출혈, 호흡곤란 등)
- medium : 24시간 내 병원 방문 권고
- low    : 며칠 내 병원 방문 권고
- observe: 가정 관찰 가능

증상: {symptoms}"""


async def urgency_judge(symptom_text: str) -> dict:
    """증상 텍스트를 분석해 긴급도(high/medium/low/observe)를 반환합니다."""
    # 규칙 기반 빠른 판단
    text_lower = symptom_text.lower()
    for kw in _EMERGENCY_KEYWORDS:
        if kw in text_lower:
            return {"type": "urgency", "urgency": "high", "method": "rule"}

    # LLM 기반 판단
    try:
        from app.core.config import settings
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from pydantic import SecretStr
        import json

        llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=SecretStr(settings.OPENAI_API_KEY),
        )
        prompt = _URGENCY_PROMPT.format(symptoms=symptom_text)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, llm.invoke, [HumanMessage(content=prompt)]
        )
        content = response.content.strip()
        # JSON 블록 추출
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        parsed = json.loads(content)
        urgency = parsed.get("urgency", "observe")
    except Exception:
        urgency = "observe"

    return {"type": "urgency", "urgency": urgency, "method": "llm"}


# ---------------------------------------------------------------------------
# 병렬 실행 진입점
# ---------------------------------------------------------------------------


async def run_parallel_analysis(
    symptom_text: str,
    breed: Optional[str] = None,
) -> dict:
    """3개의 서브에이전트를 asyncio.gather로 병렬 실행하고 결과를 합칩니다."""
    results = await asyncio.gather(
        symptom_analyzer(symptom_text),
        breed_researcher(breed),
        urgency_judge(symptom_text),
    )

    combined: dict = {}
    for r in results:
        r_type = r.get("type", "")
        if r_type == "symptom_search":
            combined["symptom_search"] = r["result"]
        elif r_type == "breed_info":
            combined["breed_info"] = r["result"]
        elif r_type == "urgency":
            combined["urgency"] = r["urgency"]

    return combined
