"""
Opik 평가 스크립트 - 의료 AI 에이전트 평가

5단계 평가 프로세스:
1. @track 트레이싱 추가 (tools.py, medical_agent.py에 적용 완료)
2. 평가 태스크 정의
3. 데이터셋 생성 (get_or_create_dataset)
4. 메트릭 선택 (Heuristic + LLM-as-a-judge)
5. 평가 실행 (evaluate)
"""

import asyncio
import json
import os
import sys
import uuid

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv

load_dotenv()

# Opik 환경변수 설정
from app.core.config import settings

if settings.OPIK is not None:
    if settings.OPIK.URL_OVERRIDE:
        os.environ["OPIK_URL_OVERRIDE"] = settings.OPIK.URL_OVERRIDE
    if settings.OPIK.API_KEY:
        os.environ["OPIK_API_KEY"] = settings.OPIK.API_KEY
    if settings.OPIK.WORKSPACE:
        os.environ["OPIK_WORKSPACE"] = settings.OPIK.WORKSPACE
    if settings.OPIK.PROJECT:
        os.environ["OPIK_PROJECT_NAME"] = settings.OPIK.PROJECT

import opik
from opik.evaluation import evaluate
from opik.evaluation.metrics import Contains, Hallucination


# ---------------------------------------------------------------------------
# 3단계: 데이터셋 준비
# ---------------------------------------------------------------------------

def create_dataset():
    """opik_dataset.json에서 평가용 데이터셋을 생성합니다."""
    client = opik.Opik()

    dataset = client.get_or_create_dataset(name="yjkim-dataset")

    # 이미 항목이 있으면 insert 스킵
    if len(dataset.get_items()) > 0:
        print(f"기존 데이터셋 사용: {len(dataset.get_items())}개 항목")
        return dataset

    # 데이터셋 파일 로드
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "app", "agents", "data", "opik_dataset.json"
    )
    with open(data_path, "r", encoding="utf-8") as f:
        patient_records = json.load(f)

    # 환자 레코드를 평가용 데이터셋 항목으로 변환
    items = []
    for record in patient_records:
        diagnosis = record["diagnosis"]
        age = record["age"]
        gender = record["gender"]
        prescriptions = ", ".join(record.get("prescriptions", []))
        tests = ", ".join(
            [f'{t["type"]}: {t["result"]}' for t in record.get("tests", [])]
        )

        # 에이전트에 전달할 질문 생성
        input_query = (
            f"{age}세 {gender} 환자입니다. "
            f"{diagnosis} 진단을 받았습니다. "
            f"이 질환에 대한 의료 정보와 주의사항을 알려주세요."
        )

        # 컨텍스트: 환자 기록에서 추출한 근거 정보
        context = [
            f"진단: {diagnosis}",
            f"검사 결과: {tests}",
            f"처방: {prescriptions}",
        ]

        items.append(
            {
                "input": input_query,
                "reference": diagnosis,
                "context": context,
                "patient_id": record["patient_id"],
            }
        )

    dataset.insert(items)
    print(f"데이터셋 준비 완료: {len(items)}개 항목")
    return dataset


# ---------------------------------------------------------------------------
# 2단계: 평가 태스크 정의
# ---------------------------------------------------------------------------

def evaluation_task(dataset_item: dict) -> dict:
    """
    데이터셋 항목을 받아 에이전트를 실행하고 메트릭이 기대하는 형식으로 반환합니다.

    Returns:
        dict with keys: output, input, context, reference
    """
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from pydantic import SecretStr

    from app.agents.medical_agent import create_medical_agent

    # 평가용 에이전트 생성 (매 호출마다 독립적인 컨텍스트)
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
    )
    agent = create_medical_agent(model=model, checkpointer=MemorySaver())

    user_query = dataset_item["input"]
    thread_id = str(uuid.uuid4())

    async def _run():
        result_content = ""
        async for chunk in agent.astream(
            {"messages": [HumanMessage(content=user_query)]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode="updates",
        ):
            for step, event in chunk.items():
                if step != "model":
                    continue
                messages = event.get("messages", [])
                if not messages:
                    continue
                msg = messages[0]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_call = msg.tool_calls[0]
                    if tool_call.get("name") == "ChatResponse":
                        result_content = tool_call.get("args", {}).get(
                            "content", ""
                        )
        return result_content

    output = asyncio.run(_run())

    return {
        "output": output,
        "input": user_query,
        "context": dataset_item.get("context", []),
        "reference": dataset_item.get("reference", ""),
    }


# ---------------------------------------------------------------------------
# 4단계: 메트릭 선택
# ---------------------------------------------------------------------------

# Heuristic 메트릭: 에이전트 응답에 진단명이 포함되어 있는지 확인
contains_metric = Contains(case_sensitive=False, name="diagnosis_mentioned")

# LLM-as-a-judge 메트릭: 에이전트가 허위 정보를 생성하지 않았는지 판단
hallucination_metric = Hallucination(name="hallucination_check")


# ---------------------------------------------------------------------------
# 5단계: 평가 실행
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("의료 AI 에이전트 평가 시작")
    print("=" * 60)

    # 데이터셋 생성/로드
    dataset = create_dataset()

    # 평가 실행
    result = evaluate(
        dataset=dataset,
        task=evaluation_task,
        scoring_metrics=[contains_metric, hallucination_metric],
        scoring_key_mapping={
            "reference": "reference",
            "input": "input",
            "context": "context",
        },
        experiment_name="yjkim-medical-agent-eval",
        task_threads=4,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("평가 완료")
    print("=" * 60)
    print(f"실험 이름: {result.experiment_name}")
    print(f"테스트 케이스 수: {len(result.test_results)}")


if __name__ == "__main__":
    main()
