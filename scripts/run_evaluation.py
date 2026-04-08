"""
강아지 증상 분석 에이전트 Opik 평가 스크립트

평가 지표:
1. answer_relevance        — 답변이 증상 질문과 관련 있는가 (LLM-as-a-judge)
2. hallucination           — 존재하지 않는 약/치료법을 언급했는가 (LLM-as-a-judge)
3. urgency_accuracy        — 긴급도 키워드가 답변에 포함됐는가 (Heuristic)
4. tool_call_accuracy      — 기대 도구를 모두 호출했는가 (Heuristic)
5. report_format_compliance — 증상 있을 때 리포트 형식으로 답변했는가 (Heuristic)
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
    base_metric,
    score_result,
)
from opik import track

from app.services.agent_service import AgentService

# ---------------------------------------------------------------------------
# 커스텀 메트릭
# ---------------------------------------------------------------------------

class UrgencyAccuracy(base_metric.BaseMetric):
    """긴급도 분류 정확도 — expected_urgency 키워드가 답변에 있으면 1.0, 없으면 0.0."""

    _KEYWORDS = {
        "high":    ["즉시 병원", "응급", "즉각", "즉시 동물병원"],
        "medium":  ["24시간", "며칠 내", "빠른", "조속히", "신속히"],
        "low":     ["가정 관찰", "천천히", "지켜보", "며칠 내", "천천히 관찰"],
        "observe": ["가정 관찰", "지켜보"],
    }

    def __init__(self):
        super().__init__(name="urgency_accuracy")

    def score(self, output: str, expected_urgency: str = "", **kwargs):
        keywords = self._KEYWORDS.get(expected_urgency, [])
        matched = any(kw in output for kw in keywords)
        return score_result.ScoreResult(
            name="urgency_accuracy",
            value=1.0 if matched else 0.0,
            reason=f"expected={expected_urgency}, matched={matched}",
        )


class ToolCallAccuracy(base_metric.BaseMetric):
    """도구 호출 정확도 — expected_tools가 모두 호출됐으면 1.0, 하나라도 누락되면 0.0."""

    def __init__(self):
        super().__init__(name="tool_call_accuracy")

    def score(self, tools_used: list, expected_tools: list, **kwargs):
        if not expected_tools:
            return score_result.ScoreResult(
                name="tool_call_accuracy",
                value=1.0,
                reason="expected_tools 없음 — 패스",
            )
        missing = [t for t in expected_tools if t not in tools_used]
        matched = len(missing) == 0
        return score_result.ScoreResult(
            name="tool_call_accuracy",
            value=1.0 if matched else 0.0,
            reason=f"expected={expected_tools}, used={tools_used}, missing={missing}",
        )



# ---------------------------------------------------------------------------
# 평가 태스크
# ---------------------------------------------------------------------------

@track(name="dog_symptom_eval_task")
async def evaluation_task(dataset_item: dict) -> dict:
    """데이터셋 항목 하나를 에이전트에 실행하고 결과를 반환한다.

    AgentService를 호출마다 새로 생성해 이벤트 루프 충돌을 방지한다.
    """
    svc = AgentService()
    thread_id = uuid.uuid4()
    full_response = ""
    tools_used: list[str] = []

    async for event in svc.process_query(dataset_item["input"], thread_id):
        try:
            parsed = json.loads(event)
            step = parsed.get("step")
            if step == "model":
                tools_used.extend(parsed.get("tool_calls", []))
            elif step == "done":
                full_response = parsed.get("content", "")
        except Exception:
            pass

    return {
        "input": dataset_item["input"],
        "output": full_response,
        "context": dataset_item.get("context", ""),
        "expected_urgency": dataset_item.get("expected_urgency", ""),
        "expected_tools": dataset_item.get("expected_tools", []),
        "tools_used": tools_used,
    }


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
        experiment_name="dog-symptom-agent-week3",
        dataset=dataset,
        task=sync_task,
        scoring_metrics=[
            AnswerRelevance(model="gpt-4.1-mini"),
            Hallucination(model="gpt-4.1-mini"),
            UrgencyAccuracy(),
            ToolCallAccuracy(),
        ],
        task_threads=1,
    )
    print("\n평가 완료. Opik 대시보드에서 결과를 확인하세요.")


if __name__ == "__main__":
    main()
