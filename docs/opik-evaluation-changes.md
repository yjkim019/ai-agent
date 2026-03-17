# Opik 평가 기능 추가 - 변경 내역

## 개요

의료 AI 에이전트에 Opik 평가 파이프라인을 추가하였습니다.
Opik의 5단계 평가 프로세스(트레이싱 → 태스크 정의 → 데이터셋 → 메트릭 → 평가 실행)를 적용합니다.

---

## 변경 파일 목록

| 파일 | 유형 | 설명 |
|------|------|------|
| `app/agents/tools.py` | 수정 | 3개 도구 함수에 `@track` 데코레이터 추가 |
| `app/agents/medical_agent.py` | 수정 | 에이전트 팩토리 함수에 `@track` 데코레이터 추가 |
| `scripts/run_evaluation.py` | 신규 | Opik 평가 스크립트 |

---

## 1단계: @track 트레이싱 추가

### `app/agents/tools.py`

`opik.track` 데코레이터를 import하고, 3개 도구 함수에 적용하였습니다.

```python
from opik import track

@tool
@track(name="search_symptoms")
def search_symptoms(symptoms: str) -> str:
    ...

@tool
@track(name="get_medication_info")
def get_medication_info(medication_name: str) -> str:
    ...

@tool
@track(name="find_nearby_hospitals")
def find_nearby_hospitals(location: str, specialty: str = "일반") -> str:
    ...
```

### `app/agents/medical_agent.py`

에이전트 생성 팩토리 함수에 `@track` 데코레이터를 적용하였습니다.

```python
from opik import track

@track(name="create_medical_agent")
def create_medical_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None):
    ...
```

---

## 2단계: 평가 태스크 정의

`scripts/run_evaluation.py`의 `evaluation_task()` 함수에서 구현합니다.

- 데이터셋 항목의 `input`(질문)을 에이전트에 전달
- 에이전트의 비동기 스트리밍(`astream`)을 `asyncio.run()`으로 동기 래핑
- 매 호출마다 `MemorySaver()`로 독립적인 에이전트 생성 (평가 간 간섭 방지)
- `ChatResponse` tool_call에서 `content`를 추출하여 반환

```python
def evaluation_task(dataset_item: dict) -> dict:
    # 에이전트 생성 및 실행
    ...
    output = asyncio.run(_run())
    return {
        "output": output,
        "input": user_query,
        "context": dataset_item.get("context", []),
        "reference": dataset_item.get("reference", ""),
    }
```

---

## 3단계: 데이터셋 생성

`app/agents/data/opik_dataset.json`의 30개 환자 레코드를 Opik 데이터셋으로 변환합니다.

- **데이터셋 이름**: `yjkim-dataset`
- **변환 방식**: 환자 레코드 → 자연어 질문 + 컨텍스트 + 기대값

```python
client = opik.Opik()
dataset = client.get_or_create_dataset(name="yjkim-dataset")
```

각 항목의 구조:

| 키 | 내용 | 예시 |
|----|------|------|
| `input` | 에이전트에 전달할 질문 | "35세 여 환자입니다. 감기 진단을 받았습니다..." |
| `reference` | 기대 진단명 (Contains 메트릭용) | "감기" |
| `context` | 근거 정보 리스트 (Hallucination 메트릭용) | ["진단: 감기", "검사 결과: 혈액검사: 정상", "처방: 해열제, 진해제"] |
| `patient_id` | 환자 식별자 | "OPIK0001" |

---

## 4단계: 메트릭 선택

두 종류의 메트릭을 사용합니다.

### Heuristic 메트릭: Contains

- **이름**: `diagnosis_mentioned`
- **목적**: 에이전트 응답에 진단명이 포함되어 있는지 확인
- **판정**: 대소문자 무시 (`case_sensitive=False`)

### LLM-as-a-judge 메트릭: Hallucination

- **이름**: `hallucination_check`
- **목적**: 에이전트가 컨텍스트에 근거하지 않은 허위 정보를 생성했는지 판단
- **입력 키**: `input`, `output`, `context`

---

## 5단계: 평가 실행

```python
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
```

### 실행 방법

```bash
cd /home/yeonju/agent
python scripts/run_evaluation.py
```

### 평가 결과 (첫 실행)

| 메트릭 | 평균 점수 | 설명 |
|--------|-----------|------|
| diagnosis_mentioned (Contains) | 1.0000 | 모든 응답에 진단명 포함 |
| hallucination_check (Hallucination) | 0.7293 | 약 73% 응답이 허위 정보 없음 |

- 테스트 케이스: 30개
- 소요 시간: 약 4분 23초
- Opik 대시보드: https://opik-edu.didim365.app → `yjkim-project` 프로젝트
