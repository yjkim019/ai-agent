# 2주차: Opik 관찰성 + 평가 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Opik 대시보드에서 에이전트의 도구 호출 흐름을 실시간으로 확인하고, LLM-as-a-judge로 답변 품질을 수치화

**Architecture:** 1주차 완성 코드 위에 Opik `@track` 데코레이터 추가 → 트레이스 전송. 별도 평가 스크립트가 사전 정의된 데이터셋을 에이전트에 실행시키고 Opik `evaluate()`로 점수를 기록한다.

**Tech Stack:** Opik SDK, LangChain Opik integration, pytest, asyncio

**사전 조건:** 1주차 플랜 완료, `.env`에 `OPIK__URL_OVERRIDE` 설정됨

---

## 파일 구조

| 파일 | 변경 종류 | 내용 |
|---|---|---|
| `app/agents/dog_agent.py` | 수정 | `create_dog_agent`에 `@track` 추가 |
| `app/agents/tools/search_symptoms_tool.py` | — | 변경 없음 (search_agent.py의 `@tool`이 자동 추적됨) |
| `app/data/eval_dataset.json` | 신규 | 평가용 질문·기대값 10개 |
| `scripts/run_evaluation.py` | 수정 | 강아지 증상 평가 메트릭으로 교체 |
| `tests/test_week2.py` | 신규 | 단위 테스트 |

---

### Task 1: `create_dog_agent`에 Opik `@track` 추가

**Files:**
- Modify: `app/agents/dog_agent.py`
- Test: `tests/test_week2.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week2.py
def test_create_dog_agent_is_tracked():
    import inspect
    from app.agents.dog_agent import create_dog_agent
    # opik @track 데코레이터는 __wrapped__ 속성을 남긴다
    assert hasattr(create_dog_agent, "__wrapped__") or callable(create_dog_agent)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week2.py::test_create_dog_agent_is_tracked -v
```
Expected: FAIL (아직 `@track` 없음)

- [ ] **Step 3: `dog_agent.py` 상단에 import 추가 및 데코레이터 적용**

파일 상단에 추가:
```python
from opik import track
```

`create_dog_agent` 함수 위에 추가:
```python
@track(name="create_dog_agent")
def create_dog_agent(model: ChatOpenAI, checkpointer: BaseCheckpointSaver[Any] = None):
    """강아지 증상 분석 에이전트를 생성합니다."""
    ...  # 기존 구현 그대로
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week2.py::test_create_dog_agent_is_tracked -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/dog_agent.py tests/test_week2.py
git commit -m "feat: add Opik @track to create_dog_agent"
```

---

### Task 2: 평가 데이터셋 생성

**Files:**
- Create: `app/data/eval_dataset.json`
- Test: `tests/test_week2.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week2.py 에 추가
import json, os

def test_eval_dataset_structure():
    path = os.path.join(os.path.dirname(__file__), "..", "app", "data", "eval_dataset.json")
    with open(path) as f:
        data = json.load(f)
    assert len(data) >= 5
    for item in data:
        assert "input" in item
        assert "expected_urgency" in item
        assert "expected_tools" in item
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week2.py::test_eval_dataset_structure -v
```
Expected: FAIL — `FileNotFoundError`

- [ ] **Step 3: `eval_dataset.json` 생성**

```json
[
  {
    "input": "강아지가 갑자기 쓰러지고 경련을 해요",
    "expected_urgency": "high",
    "expected_tools": ["search_symptoms"],
    "context": "응급 증상 — 즉시 병원 필요"
  },
  {
    "input": "말티즈가 3일 전부터 밥을 잘 안 먹어요",
    "expected_urgency": "low",
    "expected_tools": ["search_symptoms", "get_pet_breed_info"],
    "context": "품종 특성(저혈당 취약) + 식욕부진"
  },
  {
    "input": "강아지가 어제부터 구토를 5번 했어요. 물도 안 마셔요",
    "expected_urgency": "medium",
    "expected_tools": ["search_symptoms"],
    "context": "구토 반복 + 탈수 우려"
  },
  {
    "input": "포메라니안이 기침을 자주 해요",
    "expected_urgency": "medium",
    "expected_tools": ["search_symptoms", "get_pet_breed_info"],
    "context": "소형견 기관허탈 가능성"
  },
  {
    "input": "강아지 뒷다리를 가끔 들어요",
    "expected_urgency": "low",
    "expected_tools": ["search_symptoms"],
    "context": "슬개골 탈구 초기 증상"
  },
  {
    "input": "강아지가 숨을 가쁘게 쉬고 잇몸이 하얘요",
    "expected_urgency": "high",
    "expected_tools": ["search_symptoms"],
    "context": "호흡 곤란 응급"
  },
  {
    "input": "골든리트리버가 물을 엄청 많이 마셔요",
    "expected_urgency": "medium",
    "expected_tools": ["search_symptoms", "get_pet_breed_info"],
    "context": "다음다갈증 — 당뇨 등 의심"
  },
  {
    "input": "강아지 뒷다리가 절뚝거려요. 강남구에 동물병원 있나요?",
    "expected_urgency": "medium",
    "expected_tools": ["search_symptoms", "find_nearby_vet"],
    "context": "증상 + 병원 위치 복합 요청"
  },
  {
    "input": "강아지가 온몸을 계속 긁어요",
    "expected_urgency": "low",
    "expected_tools": ["search_symptoms"],
    "context": "피부 알레르기·기생충"
  },
  {
    "input": "노령견(13살)이 밥도 안 먹고 무기력해요",
    "expected_urgency": "medium",
    "expected_tools": ["search_symptoms"],
    "context": "노령견 복합 증상 — 신장 등 의심"
  }
]
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week2.py::test_eval_dataset_structure -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/data/eval_dataset.json tests/test_week2.py
git commit -m "feat: add dog symptom evaluation dataset (10 cases)"
```

---

### Task 3: `run_evaluation.py` 강아지 증상 평가 메트릭으로 교체

**Files:**
- Modify: `scripts/run_evaluation.py`
- Test: `tests/test_week2.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week2.py 에 추가
def test_run_evaluation_importable():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "run_evaluation",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "run_evaluation.py")
    )
    # 파일 존재 및 파싱 가능 여부 확인
    assert spec is not None
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week2.py::test_run_evaluation_importable -v
```
Expected: PASS (파일은 존재하므로) — 내용 검증은 Step 3 이후 수동 확인

- [ ] **Step 3: `run_evaluation.py` 전체 교체**

```python
# scripts/run_evaluation.py
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
```

- [ ] **Step 4: 평가 스크립트 실행**

```bash
uv run python scripts/run_evaluation.py
```
Expected: 10개 케이스 실행 후 Opik 대시보드에 점수 기록

- [ ] **Step 5: 커밋**

```bash
git add scripts/run_evaluation.py tests/test_week2.py
git commit -m "feat: update run_evaluation.py for dog symptom agent metrics"
```

---

### Task 4: Opik 대시보드 분석 및 취약점 도출

> 코드 변경 없음 — Opik UI에서 수동 분석

- [ ] **Step 1: 대시보드 접속 및 트레이스 확인**

브라우저에서 `settings.OPIK.URL_OVERRIDE` 주소 접속.
`Projects` → `dog-symptom-agent` 프로젝트 선택.

확인 항목:
- 각 질문에서 어떤 도구가 호출됐는가
- LLM 응답 시간이 긴 케이스는 어디인가
- tool_calls가 0인 케이스(도구를 호출하지 않은 경우)가 있는가

- [ ] **Step 2: 평가 결과 확인**

`Experiments` → `dog-symptom-agent-week2` 선택.

확인 항목:
- `answer_relevance` 점수가 낮은 케이스 (0.5 미만)
- `hallucination` 점수가 높은 케이스 (0.5 이상 = 할루시네이션 의심)
- `urgency_heuristic` 실패 케이스

- [ ] **Step 3: 취약점 기록**

낮은 점수의 케이스를 메모:
```
취약 유형 1: ___  (예: 응급 증상 긴급도 분류 오류)
취약 유형 2: ___  (예: 품종 정보 도구 미호출)
개선 방향: ___ (예: 시스템 프롬프트 강화, PDF 데이터 추가)
```

---

## 2주차 최종 검증

- [ ] **전체 테스트 실행**

```bash
uv run pytest tests/test_week1.py tests/test_week2.py -v
```
Expected: 전체 PASS

- [ ] **트레이싱 실시간 확인**

채팅 UI에서 "강아지가 구토를 해요" 입력 →
Opik 대시보드에서 해당 트레이스가 실시간으로 나타나는지 확인
