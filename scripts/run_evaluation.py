"""
강아지 증상 분석 에이전트 Opik 평가 스크립트

평가 지표:
1. answer_relevance   — 답변이 증상 질문과 관련 있는가 (LLM-as-a-judge)
2. urgency_accuracy   — 긴급도 분류가 기대값과 일치하는가 (Heuristic)
3. hallucination      — 존재하지 않는 약/치료법을 언급했는가 (LLM-as-a-judge)
"""
import asyncio, json, os, sys, uuid
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings

if settings.OPIK is not None and settings.OPIK.URL_OVERRIDE:
    os.environ["OPIK_URL_OVERRIDE"] = settings.OPIK.URL_OVERRIDE
if settings.OPIK is not None and settings.OPIK.API_KEY:
    os.environ["OPIK_API_KEY"] = settings.OPIK.API_KEY

import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import (
    AnswerRelevance,
    Hallucination,
)
from opik import track

from app.services.agent_service import AgentService

# ---------------------------------------------------------------------------
# 평가 태스크
# ---------------------------------------------------------------------------

agent_service = AgentService()

@track(name="dog_symptom_eval_task")
async def evaluation_task(dataset_item: dict) -> dict:
    """데이터셋 항목 하나를 에이전트에 실행하고 결과를 반환한다."""
    thread_id = uuid.uuid4()
    full_response = ""
    async for event in agent_service.process_query(dataset_item["input"], thread_id):
        import json as _json
        try:
            parsed = _json.loads(event)
            if parsed.get("step") == "done":
                full_response = parsed.get("content", "")
        except Exception:
            pass
    return {
        "input": dataset_item["input"],
        "output": full_response,
        "context": dataset_item.get("context", ""),
        "expected_urgency": dataset_item.get("expected_urgency", ""),
    }


def urgency_heuristic(dataset_item: dict, llm_output: dict) -> float:
    """긴급도 분류 정확도 — 기대값과 일치하면 1.0, 아니면 0.0."""
    urgency_keywords = {
        "high": ["즉시 병원", "응급", "즉각"],
        "medium": ["24시간", "며칠 내", "빠른"],
        "low": ["가정 관찰", "천천히", "지켜보"],
        "observe": ["가정 관찰"],
    }
    expected = dataset_item.get("expected_urgency", "")
    output = llm_output.get("output", "").lower()
    keywords = urgency_keywords.get(expected, [])
    return 1.0 if any(kw in output for kw in keywords) else 0.0


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------

def main():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "eval_dataset.json")
    with open(dataset_path) as f:
        raw_dataset = json.load(f)

    # Opik 데이터셋 등록
    client = opik.Opik()
    dataset = client.get_or_create_dataset(name="dog-symptom-eval-v1")
    dataset.insert(raw_dataset)

    # 동기 래퍼
    def sync_task(item):
        return asyncio.run(evaluation_task(item))

    evaluate(
        experiment_name="dog-symptom-agent-week2",
        dataset=dataset,
        task=sync_task,
        scoring_metrics=[
            AnswerRelevance(model="gpt-4o"),
            Hallucination(model="gpt-4o"),
        ],
        task_threads=1,
    )
    print("\n평가 완료. Opik 대시보드에서 결과를 확인하세요.")


if __name__ == "__main__":
    main()
