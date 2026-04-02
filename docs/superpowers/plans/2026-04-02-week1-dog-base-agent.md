# 1주차: 강아지 증상 분석 챗봇 기초 에이전트 구현 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 채팅 UI에서 강아지 증상을 입력하면 기초 분석 답변이 나오는 웹서비스 완성

**Architecture:** 기존 `pet_agent.py` 구조를 `dog_agent.py`로 전환. `create_agent` 오케스트레이터 + `search_symptoms`(BM25) + `get_pet_breed_info` + `find_nearby_vet` 도구 3개. `search_agent.py`의 StateGraph, `find_nearby_vet`, `get_pet_breed_info`를 그대로 재활용한다.

**Tech Stack:** FastAPI, LangGraph `create_agent`, Elasticsearch BM25, LangChain `@tool`, pytest

---

## 파일 구조

| 파일 | 변경 종류 | 내용 |
|---|---|---|
| `app/agents/tools/_es_common.py` | 수정 | 인덱스명 `dog-symptoms`, `dog-knowledge`로 변경 |
| `app/agents/search_agent.py` | 수정 | `search` → `search_symptoms` 리네임 및 설명 업데이트 |
| `app/agents/pet_agent.py` | 수정 | 리네임된 `search_symptoms` 임포트 반영 |
| `app/agents/prompts.py` | 수정 | `DOG_SYMPTOM_SYSTEM_PROMPT` 추가 |
| `app/agents/dog_agent.py` | 신규 | `create_dog_agent()` 팩토리 |
| `app/services/agent_service.py` | 수정 | `create_dog_agent` 사용, Opik 태그 변경 |
| `app/main.py` | 수정 | 앱 타이틀 변경 |
| `scripts/seed_dog_symptoms.py` | 신규 | ES `dog-symptoms` 인덱스 샘플 데이터 적재 (Week 1 기능 검증용) |
| `tests/test_week1.py` | 신규 | 단위 테스트 |

> **샘플 데이터 이유:** 3주차 PDF 파이프라인 전까지 `search_symptoms` 도구가 빈 결과를 반환하지 않도록
> 10개의 수동 작성 증상 문서를 임시로 적재한다. 3주차에 실제 PDF 데이터로 교체된다.

---

### Task 1: ES 인덱스 이름 변경

**Files:**
- Modify: `app/agents/tools/_es_common.py:22-25`
- Test: `tests/test_week1.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week1.py
def test_es_index_names():
    from app.agents.tools._es_common import INDEX_NAME, BM25_INDEX_NAME
    assert INDEX_NAME == "dog-knowledge"
    assert BM25_INDEX_NAME == "dog-symptoms"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week1.py::test_es_index_names -v
```
Expected: FAIL — `AssertionError: assert 'pet-knowledge' == 'dog-knowledge'`

- [ ] **Step 3: `_es_common.py` 22~25번째 줄 수정**

```python
# Vector / Hybrid 검색용 인덱스
INDEX_NAME = "dog-knowledge"
# BM25 키워드 검색용 인덱스
BM25_INDEX_NAME = "dog-symptoms"
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week1.py::test_es_index_names -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/tools/_es_common.py tests/test_week1.py
git commit -m "feat: update ES index names to dog-symptoms and dog-knowledge"
```

---

### Task 2: `search` 도구 → `search_symptoms` 리네임

**Files:**
- Modify: `app/agents/search_agent.py:195-201`
- Modify: `app/agents/pet_agent.py:11,52`
- Test: `tests/test_week1.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week1.py 에 추가
def test_search_symptoms_tool_name():
    from app.agents.search_agent import search_symptoms
    assert search_symptoms.name == "search_symptoms"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week1.py::test_search_symptoms_tool_name -v
```
Expected: FAIL — `ImportError: cannot import name 'search_symptoms'`

- [ ] **Step 3: `search_agent.py` 195~201번째 줄 함수 교체**

```python
@tool
def search_symptoms(search_query: str) -> str:
    """강아지 증상 및 질환 관련 의료 정보를 검색합니다.
    구토, 식욕부진, 기침, 절뚝거림 등 증상과 의심 질환 정보 검색에 사용하세요."""
    result = _search_graph.invoke({"query": search_query})
    return result["result"]
```

- [ ] **Step 4: `pet_agent.py` 임포트 수정**

11번째 줄:
```python
from app.agents.search_agent import search_symptoms
```

52번째 줄 tools 목록:
```python
tools=[search_symptoms, get_pet_breed_info, find_nearby_vet],
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
uv run pytest tests/test_week1.py::test_search_symptoms_tool_name -v
```
Expected: PASS

- [ ] **Step 6: 커밋**

```bash
git add app/agents/search_agent.py app/agents/pet_agent.py tests/test_week1.py
git commit -m "feat: rename search tool to search_symptoms"
```

---

### Task 3: `DOG_SYMPTOM_SYSTEM_PROMPT` 작성

**Files:**
- Modify: `app/agents/prompts.py`
- Test: `tests/test_week1.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week1.py 에 추가
def test_dog_symptom_prompt_keywords():
    from app.agents.prompts import DOG_SYMPTOM_SYSTEM_PROMPT
    assert "강아지" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "긴급도" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "search_symptoms" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "get_pet_breed_info" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "find_nearby_vet" in DOG_SYMPTOM_SYSTEM_PROMPT
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week1.py::test_dog_symptom_prompt_keywords -v
```
Expected: FAIL — `ImportError: cannot import name 'DOG_SYMPTOM_SYSTEM_PROMPT'`

- [ ] **Step 3: `prompts.py` 에 프롬프트 추가**

기존 `PET_SYSTEM_PROMPT` 아래에 추가:

```python
DOG_SYMPTOM_SYSTEM_PROMPT = """당신은 강아지 증상 분석 전문 AI 어시스턴트입니다.
견주가 강아지의 이상 증상을 설명하면, 단계적으로 질문하여 충분한 정보를 수집한 뒤
긴급도·의심 질환·가정 대처법·병원 방문 권고를 담은 종합 리포트를 제공합니다.

# 사용 가능한 도구:
- search_symptoms: 강아지 증상 및 질환 관련 의료 정보 검색
- get_pet_breed_info: 품종별 취약 질환 조회 (품종명은 반드시 한글로)
- find_nearby_vet: 주변 동물병원 검색

# 대화 원칙:
1. 증상이 불명확하면 추가 질문을 한다 (최대 3회):
   - "언제부터 그런 증상이 있었나요?"
   - "밥이나 물은 먹고 있나요?"
   - "다른 이상한 행동도 보이나요?"
2. 충분한 정보가 모이면 search_symptoms로 관련 질환을 검색한다.
3. 품종이 언급된 경우 get_pet_breed_info로 취약 질환을 추가 확인한다.
4. 검색 결과를 바탕으로 종합 리포트를 작성한다.

⚠️ 주의:
- 검색은 총 6회 이내로 제한한다.
- 본 답변은 수의사의 전문적인 진료를 대체하지 않습니다.
- 경련, 의식 저하, 심한 출혈 등 응급 증상은 즉시 병원 방문을 권고한다.

# Response Format:
반드시 아래 JSON 형식으로 응답하세요:
{
    "message_id": "<UUID 형식의 고유 메시지 ID>",
    "content": "**긴급도**: [즉시 병원 / 24시간 내 방문 / 며칠 내 방문 / 가정 관찰 가능]\\n\\n**의심 질환**:\\n...\\n\\n**가정 내 대처법**:\\n...\\n\\n**병원 방문 시 전달사항**:\\n...",
    "metadata": {"urgency": "high|medium|low|observe"}
}
"""
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week1.py::test_dog_symptom_prompt_keywords -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/prompts.py tests/test_week1.py
git commit -m "feat: add DOG_SYMPTOM_SYSTEM_PROMPT"
```

---

### Task 4: `dog_agent.py` 생성

**Files:**
- Create: `app/agents/dog_agent.py`
- Test: `tests/test_week1.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week1.py 에 추가
def test_create_dog_agent_importable():
    from app.agents.dog_agent import create_dog_agent
    assert callable(create_dog_agent)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week1.py::test_create_dog_agent_importable -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.agents.dog_agent'`

- [ ] **Step 3: `dog_agent.py` 생성**

```python
# app/agents/dog_agent.py
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week1.py::test_create_dog_agent_importable -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/dog_agent.py tests/test_week1.py
git commit -m "feat: create dog_agent.py with symptom analysis tools"
```

---

### Task 5: `agent_service.py` 및 `main.py` 업데이트

**Files:**
- Modify: `app/services/agent_service.py:53-59,75-82`
- Modify: `app/main.py:9-13`
- Test: `tests/test_week1.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week1.py 에 추가
def test_main_app_title_contains_dog():
    from app.main import app
    assert "강아지" in app.title
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week1.py::test_main_app_title_contains_dog -v
```
Expected: FAIL

- [ ] **Step 3: `main.py` 9~13번째 줄 교체**

```python
app = FastAPI(
    title="강아지 증상 분석 AI 챗봇",
    description="강아지 증상을 입력하면 긴급도·의심 질환·대처법을 알려주는 AI 어시스턴트",
    version="1.0.0",
)
```

- [ ] **Step 4: `agent_service.py` `_create_agent` 메서드 교체 (75~82번째 줄)**

```python
def _create_agent(self):
    """강아지 증상 분석 에이전트 생성"""
    from app.agents.dog_agent import create_dog_agent
    assert self.checkpointer is not None, "checkpointer가 초기화되지 않았습니다."
    self.agent = create_dog_agent(
        model=self.model,
        checkpointer=self.checkpointer,
    )
    if self.opik_tracer is not None:
        from opik.integrations.langchain import track_langgraph
        self.agent = track_langgraph(self.agent, self.opik_tracer)
```

- [ ] **Step 5: Opik 태그 변경 (53~59번째 줄)**

```python
self.opik_tracer = OpikTracer(
    tags=["dog-symptom-agent"],
    metadata={"model": settings.OPENAI_MODEL}
)
```

- [ ] **Step 6: 테스트 통과 확인**

```bash
uv run pytest tests/test_week1.py -v
```
Expected: 전체 PASS

- [ ] **Step 7: 커밋**

```bash
git add app/main.py app/services/agent_service.py tests/test_week1.py
git commit -m "feat: update agent_service and main.py for dog symptom chatbot"
```

---

### Task 6: 샘플 데이터 적재 스크립트

> 3주차 PDF 파이프라인 전까지 `search_symptoms`가 빈 결과를 반환하지 않도록 임시 데이터를 적재한다.

**Files:**
- Create: `scripts/seed_dog_symptoms.py`

- [ ] **Step 1: 스크립트 생성**

```python
# scripts/seed_dog_symptoms.py
"""
dog-symptoms ES 인덱스에 강아지 증상 샘플 데이터를 적재한다.
실행: uv run python scripts/seed_dog_symptoms.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from app.agents.tools._es_common import get_es_client_bm25, BM25_INDEX_NAME, CONTENT_FIELD

INDEX_SETTINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            CONTENT_FIELD: {"type": "text", "analyzer": "standard"},
            "metadata": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "keyword"},
                    "urgency": {"type": "keyword"},
                    "source": {"type": "keyword"},
                },
            },
        }
    },
}

SAMPLE_DOCS = [
    {CONTENT_FIELD: "강아지 구토: 위장염, 췌장염, 이물질 섭취가 원인입니다. 하루 3회 이상이거나 혈액이 섞인 구토, 12시간 이상 지속 시 즉시 진찰이 필요합니다.", "metadata": {"symptom": "구토", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 식욕부진: 24~48시간 이상 밥을 먹지 않으면 수의사 진찰을 권장합니다. 스트레스, 위장 질환, 신부전, 간 질환이 원인일 수 있습니다.", "metadata": {"symptom": "식욕부진", "urgency": "low", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 기침: 켄넬코프, 심장 질환, 기관허탈이 원인입니다. 소형견(포메라니안, 치와와)은 기관허탈에 취약합니다. 혈액이 섞이면 즉시 응급 진찰이 필요합니다.", "metadata": {"symptom": "기침", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 설사: 세균성·바이러스성 장염, 기생충, 음식 알레르기가 원인입니다. 혈변이나 탈수 증상이 있으면 즉시 진찰이 필요합니다.", "metadata": {"symptom": "설사", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 무기력증: 빈혈, 감염병, 심장 질환의 신호일 수 있습니다. 구토·식욕부진·창백한 잇몸과 함께 나타나면 즉시 병원이 필요합니다.", "metadata": {"symptom": "무기력증", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 경련: 뇌전증, 독소 섭취, 저혈당의 신호입니다. 5분 이상 지속되거나 하루 2회 이상 반복되면 응급 상황입니다. 즉시 동물병원 응급실로 이동하세요.", "metadata": {"symptom": "경련", "urgency": "high", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 절뚝거림: 슬개골 탈구, 관절염, 골절이 원인입니다. 소형견(푸들, 포메라니안, 말티즈)은 슬개골 탈구 발생률이 높습니다.", "metadata": {"symptom": "절뚝거림", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 피부 가려움증: 알레르기성 피부염, 아토피, 외부 기생충이 원인입니다. 탈모·발적·딱지가 함께 나타나면 수의사 진찰이 필요합니다.", "metadata": {"symptom": "가려움증", "urgency": "low", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 음수량 증가: 당뇨병, 쿠싱증후군, 신부전, 자궁축농증의 신호일 수 있습니다. 중년~노령 암컷에서 복부 팽만이 함께 나타나면 즉시 진찰이 필요합니다.", "metadata": {"symptom": "음수량증가", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 호흡 곤란: 잇몸이 파란색/흰색이거나 입을 벌리고 숨쉬면 심장 질환, 폐렴, 기흉의 응급 신호입니다. 즉시 동물병원 응급실로 이동하세요.", "metadata": {"symptom": "호흡곤란", "urgency": "high", "source": "dog-symptom-guide"}},
]


def seed():
    es = get_es_client_bm25()
    if not es.indices.exists(index=BM25_INDEX_NAME):
        es.indices.create(index=BM25_INDEX_NAME, body=INDEX_SETTINGS)
        print(f"인덱스 생성: {BM25_INDEX_NAME}")
    else:
        print(f"인덱스 이미 존재: {BM25_INDEX_NAME}")
    for i, doc in enumerate(SAMPLE_DOCS):
        es.index(index=BM25_INDEX_NAME, id=str(i + 1), document=doc)
        print(f"  [{i+1}/{len(SAMPLE_DOCS)}] {doc['metadata']['symptom']} 적재")
    es.indices.refresh(index=BM25_INDEX_NAME)
    count = es.count(index=BM25_INDEX_NAME)["count"]
    print(f"\n완료: 총 {count}개 문서")


if __name__ == "__main__":
    seed()
```

- [ ] **Step 2: 스크립트 실행**

```bash
uv run python scripts/seed_dog_symptoms.py
```
Expected:
```
인덱스 생성: dog-symptoms
  [1/10] 구토 적재
  ...
완료: 총 10개 문서
```

- [ ] **Step 3: 검색 동작 확인**

```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from app.agents.tools._es_common import get_es_client_bm25, BM25_INDEX_NAME
es = get_es_client_bm25()
r = es.search(index=BM25_INDEX_NAME, body={'query': {'match': {'content': '구토'}}})
print(f'구토 검색 결과: {len(r[\"hits\"][\"hits\"])}건')
print(r['hits']['hits'][0]['_source']['content'][:80])
"
```
Expected: `구토 검색 결과: 1건` 이상

- [ ] **Step 4: 커밋**

```bash
git add scripts/seed_dog_symptoms.py
git commit -m "feat: add dog-symptoms ES seed script with 10 sample documents"
```

---

## 1주차 최종 검증

- [ ] **서버 구동 확인**

```bash
uv run uvicorn app.main:app --reload --port 8000
```
`http://localhost:8000/docs` → 타이틀 "강아지 증상 분석 AI 챗봇" 확인

- [ ] **채팅 시나리오 테스트**

채팅 UI에서 아래 메시지 전송:
```
강아지가 어제부터 구토를 계속 해요
```
Expected: 에이전트가 추가 질문을 하거나 증상 분석 리포트 반환
