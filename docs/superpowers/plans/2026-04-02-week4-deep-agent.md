# 4주차: Deep Agent — 서브에이전트 병렬 분석 + 장기 메모리 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 복합 증상을 서브에이전트가 병렬로 분석하고, 반려견 프로필(품종·나이·진단 이력)을 기억하는 지능형 진단 서비스 완성

**Architecture:** 3주차 `SymptomState` StateGraph에 `pet_profile` 상태 추가. 증상이 3개 이상이면 `task()`로 3개 서브에이전트(증상분석·품종리서치·응급판단)에 병렬 위임 → VFS(로컬 JSON 파일)에 중간 결과 저장 → 오케스트레이터가 병합해 최종 리포트 생성.

**Tech Stack:** LangGraph `StateGraph`, `asyncio.gather` 병렬 실행, 로컬 JSON VFS, SQLite 장기 메모리

**사전 조건:** 3주차 플랜 완료, `build_symptom_graph` 동작 확인됨

---

## 파일 구조

| 파일 | 변경 종류 | 내용 |
|---|---|---|
| `app/agents/subagents/symptom_analyzer.py` | 신규 | RAG 검색 서브에이전트 |
| `app/agents/subagents/breed_researcher.py` | 신규 | 품종 리스크 서브에이전트 |
| `app/agents/subagents/urgency_judge.py` | 신규 | 긴급도 점수 서브에이전트 |
| `app/agents/subagents/__init__.py` | 신규 | 패키지 초기화 |
| `app/services/vfs_service.py` | 신규 | VFS(로컬 JSON) 읽기/쓰기 서비스 |
| `app/agents/symptom_pipeline.py` | 수정 | `pet_profile` 상태 + 복잡도 분기 + 병렬 위임 추가 |
| `tests/test_week4.py` | 신규 | 단위 테스트 |

---

### Task 1: VFS 서비스 구현

**Files:**
- Create: `app/services/vfs_service.py`
- Test: `tests/test_week4.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week4.py
import os, json, tempfile

def test_vfs_write_and_read():
    from app.services.vfs_service import VFSService
    with tempfile.TemporaryDirectory() as tmpdir:
        vfs = VFSService(base_dir=tmpdir)
        vfs.write("symptom_analysis.json", {"result": "구토 — 위장염 의심"})
        data = vfs.read("symptom_analysis.json")
        assert data["result"] == "구토 — 위장염 의심"

def test_vfs_clear_removes_all_files():
    from app.services.vfs_service import VFSService
    with tempfile.TemporaryDirectory() as tmpdir:
        vfs = VFSService(base_dir=tmpdir)
        vfs.write("a.json", {"x": 1})
        vfs.write("b.json", {"y": 2})
        vfs.clear()
        assert vfs.read("a.json") is None
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week4.py::test_vfs_write_and_read tests/test_week4.py::test_vfs_clear_removes_all_files -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `vfs_service.py` 생성**

```python
# app/services/vfs_service.py
"""Virtual File System 서비스 — 서브에이전트 중간 결과를 로컬 JSON 파일로 저장한다."""
from __future__ import annotations

import json
import os


class VFSService:
    """thread_id 별로 격리된 디렉터리에 JSON 파일을 읽고 쓴다."""

    def __init__(self, base_dir: str = "/tmp/dog_agent_vfs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

    def write(self, filename: str, data: dict) -> None:
        with open(self._path(filename), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def read(self, filename: str) -> dict | None:
        path = self._path(filename)
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def clear(self) -> None:
        for fname in os.listdir(self.base_dir):
            if fname.endswith(".json"):
                os.remove(self._path(fname))
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week4.py::test_vfs_write_and_read tests/test_week4.py::test_vfs_clear_removes_all_files -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/services/vfs_service.py tests/test_week4.py
git commit -m "feat: add VFSService for subagent intermediate results"
```

---

### Task 2: 서브에이전트 3개 구현

**Files:**
- Create: `app/agents/subagents/__init__.py`
- Create: `app/agents/subagents/symptom_analyzer.py`
- Create: `app/agents/subagents/breed_researcher.py`
- Create: `app/agents/subagents/urgency_judge.py`
- Test: `tests/test_week4.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week4.py 에 추가
def test_subagents_importable():
    from app.agents.subagents.symptom_analyzer import analyze_symptoms
    from app.agents.subagents.breed_researcher import research_breed_risks
    from app.agents.subagents.urgency_judge import judge_urgency
    assert callable(analyze_symptoms)
    assert callable(research_breed_risks)
    assert callable(judge_urgency)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week4.py::test_subagents_importable -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: 패키지 초기화 파일 생성**

```python
# app/agents/subagents/__init__.py
from app.agents.subagents.symptom_analyzer import analyze_symptoms
from app.agents.subagents.breed_researcher import research_breed_risks
from app.agents.subagents.urgency_judge import judge_urgency

__all__ = ["analyze_symptoms", "research_breed_risks", "judge_urgency"]
```

- [ ] **Step 4: `symptom_analyzer.py` 생성**

```python
# app/agents/subagents/symptom_analyzer.py
"""RAG 검색으로 증상-질환 매핑을 분석하는 서브에이전트."""
from __future__ import annotations

from app.agents.search_agent import search_symptoms


async def analyze_symptoms(symptom_text: str) -> dict:
    """증상 텍스트로 RAG 검색을 실행하고 의심 질환 목록을 반환한다."""
    result = search_symptoms.invoke({"search_query": symptom_text})
    return {"type": "symptom_analysis", "raw": result, "query": symptom_text}
```

- [ ] **Step 5: `breed_researcher.py` 생성**

```python
# app/agents/subagents/breed_researcher.py
"""품종별 취약 질환을 조회하는 서브에이전트."""
from __future__ import annotations

from app.agents.tools import get_pet_breed_info


async def research_breed_risks(breed_name: str) -> dict:
    """품종명으로 취약 질환 정보를 조회한다. 품종 정보가 없으면 빈 결과 반환."""
    if not breed_name:
        return {"type": "breed_research", "raw": "", "breed": ""}
    result = get_pet_breed_info.invoke({"breed_name": breed_name})
    return {"type": "breed_research", "raw": result, "breed": breed_name}
```

- [ ] **Step 6: `urgency_judge.py` 생성**

```python
# app/agents/subagents/urgency_judge.py
"""증상 심각도를 규칙 기반으로 점수화하는 서브에이전트."""
from __future__ import annotations

# 긴급 키워드 → 즉시 병원
HIGH_URGENCY_KEYWORDS = ["경련", "발작", "쓰러", "호흡곤란", "잇몸 파란", "잇몸 흰", "의식 없", "대량 출혈"]
# 중간 키워드 → 24시간 내 방문
MEDIUM_URGENCY_KEYWORDS = ["구토", "설사", "식욕부진", "무기력", "절뚝", "기침", "혈변", "혈뇨"]


async def judge_urgency(symptom_text: str) -> dict:
    """증상 텍스트에서 긴급도를 판단한다."""
    text = symptom_text.lower()
    if any(kw in text for kw in HIGH_URGENCY_KEYWORDS):
        level, label = "high", "즉시 병원"
    elif any(kw in text for kw in MEDIUM_URGENCY_KEYWORDS):
        level, label = "medium", "24시간 내 방문 권장"
    else:
        level, label = "low", "가정 관찰 가능"
    return {"type": "urgency_judge", "level": level, "label": label}
```

- [ ] **Step 7: 테스트 통과 확인**

```bash
uv run pytest tests/test_week4.py::test_subagents_importable -v
```
Expected: PASS

- [ ] **Step 8: 커밋**

```bash
git add app/agents/subagents/ tests/test_week4.py
git commit -m "feat: add three subagents (symptom_analyzer, breed_researcher, urgency_judge)"
```

---

### Task 3: `symptom_pipeline.py` 복잡도 분기 + 병렬 위임 추가

> 증상이 단순(2개 이하)이면 기존 3주차 흐름 유지.
> 증잡(3개 이상)이면 3개 서브에이전트를 asyncio.gather로 병렬 실행 → VFS에 저장 → 병합 리포트.

**Files:**
- Modify: `app/agents/symptom_pipeline.py`
- Test: `tests/test_week4.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week4.py 에 추가
import asyncio

def test_is_complex_case_detects_multiple_symptoms():
    from app.agents.symptom_pipeline import _is_complex_case
    assert _is_complex_case("구토하고 설사하고 무기력해요") is True
    assert _is_complex_case("구토해요") is False
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week4.py::test_is_complex_case_detects_multiple_symptoms -v
```
Expected: FAIL

- [ ] **Step 3: `symptom_pipeline.py`에 복잡도 판단 함수 추가**

파일 최상단 import 아래에 추가:

```python
import asyncio
from app.agents.subagents import analyze_symptoms, research_breed_risks, judge_urgency
from app.services.vfs_service import VFSService

_COMPLEX_SYMPTOM_KEYWORDS = [
    "구토", "설사", "기침", "절뚝", "무기력", "식욕부진",
    "경련", "호흡", "가려움", "피부", "눈물",
]


def _is_complex_case(symptom_text: str) -> bool:
    """증상 텍스트에서 키워드가 3개 이상 등장하면 복잡한 케이스로 판단한다."""
    count = sum(1 for kw in _COMPLEX_SYMPTOM_KEYWORDS if kw in symptom_text)
    return count >= 3
```

- [ ] **Step 4: `SymptomState`에 `pet_profile` 추가**

```python
class SymptomState(TypedDict):
    messages: Annotated[list, add_messages]
    question_count: int
    pet_profile: dict   # {"name": "", "breed": "", "age": "", "history": []}
```

- [ ] **Step 5: `route_after_collect`에 복잡도 분기 추가**

```python
def route_after_collect(state: SymptomState) -> Literal["ask_follow_up", "search_rag", "parallel_analysis"]:
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    if state["question_count"] >= 3 or _has_enough_info(state):
        if _is_complex_case(user_content):
            return "parallel_analysis"
        return "search_rag"
    return "ask_follow_up"
```

- [ ] **Step 6: `parallel_analysis` 노드 추가**

```python
def parallel_analysis(state: SymptomState) -> dict:
    """3개 서브에이전트를 병렬로 실행하고 VFS에 중간 결과를 저장한다."""
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    breed = state.get("pet_profile", {}).get("breed", "")

    async def _run():
        return await asyncio.gather(
            analyze_symptoms(user_content),
            research_breed_risks(breed),
            judge_urgency(user_content),
        )

    symptom_result, breed_result, urgency_result = asyncio.run(_run())

    vfs = VFSService()
    vfs.clear()
    vfs.write("symptom_analysis.json", symptom_result)
    vfs.write("breed_risk.json", breed_result)
    vfs.write("urgency_score.json", urgency_result)

    # 병합 컨텍스트를 SystemMessage로 삽입
    merged_context = (
        f"[증상 분석]\n{symptom_result['raw']}\n\n"
        f"[품종 리스크]\n{breed_result['raw']}\n\n"
        f"[긴급도 판단] {urgency_result['label']} ({urgency_result['level']})"
    )
    return {"messages": [SystemMessage(content=merged_context)]}
```

- [ ] **Step 7: `build_symptom_graph`에 `parallel_analysis` 노드와 엣지 추가**

```python
def build_symptom_graph(checkpointer=None):
    builder = StateGraph(SymptomState)

    builder.add_node("collect_symptoms", collect_symptoms)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("search_rag", search_rag)
    builder.add_node("parallel_analysis", parallel_analysis)   # 신규
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "collect_symptoms")
    builder.add_conditional_edges(
        "collect_symptoms",
        route_after_collect,
        {
            "ask_follow_up": "ask_follow_up",
            "search_rag": "search_rag",
            "parallel_analysis": "parallel_analysis",          # 신규
        },
    )
    builder.add_edge("ask_follow_up", END)
    builder.add_edge("search_rag", "generate_report")
    builder.add_edge("parallel_analysis", "generate_report")   # 신규
    builder.add_edge("generate_report", END)

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 8: 테스트 통과 확인**

```bash
uv run pytest tests/test_week4.py::test_is_complex_case_detects_multiple_symptoms -v
```
Expected: PASS

- [ ] **Step 9: 커밋**

```bash
git add app/agents/symptom_pipeline.py tests/test_week4.py
git commit -m "feat: add parallel subagent dispatch for complex symptom cases"
```

---

### Task 4: 반려견 프로필 장기 메모리 (SQLite)

**Files:**
- Modify: `app/agents/symptom_pipeline.py`
- Test: `tests/test_week4.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week4.py 에 추가
def test_extract_pet_profile_from_message():
    from app.agents.symptom_pipeline import _extract_pet_profile
    profile = _extract_pet_profile("7살 진돗개 수컷이 구토를 해요")
    assert profile.get("breed") == "진돗개"
    assert profile.get("age") == "7살"
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week4.py::test_extract_pet_profile_from_message -v
```
Expected: FAIL

- [ ] **Step 3: `_extract_pet_profile` 함수 추가**

`symptom_pipeline.py` 유틸리티 섹션에 추가:

```python
import re

_KNOWN_BREEDS = [
    "진돗개", "말티즈", "푸들", "포메라니안", "골든리트리버",
    "래브라도", "시바이누", "비숑", "시츄", "치와와", "불독",
]


def _extract_pet_profile(text: str) -> dict:
    """메시지에서 품종, 나이를 추출한다."""
    profile: dict = {}
    for breed in _KNOWN_BREEDS:
        if breed in text:
            profile["breed"] = breed
            break
    age_match = re.search(r"(\d+)\s*살", text)
    if age_match:
        profile["age"] = age_match.group(0)
    return profile
```

- [ ] **Step 4: `collect_symptoms` 노드에 프로필 추출 추가**

```python
def collect_symptoms(state: SymptomState) -> dict:
    """첫 번째 노드: 메시지에서 반려견 프로필을 추출한다."""
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    existing_profile = state.get("pet_profile") or {}
    new_profile = _extract_pet_profile(user_content)
    merged = {**existing_profile, **new_profile}
    return {"pet_profile": merged} if merged else {}
```

- [ ] **Step 5: 테스트 통과 확인**

```bash
uv run pytest tests/test_week4.py -v
```
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add app/agents/symptom_pipeline.py tests/test_week4.py
git commit -m "feat: add pet profile extraction and long-term memory via SQLite checkpointer"
```

---

## 4주차 최종 검증

- [ ] **전체 테스트 실행**

```bash
uv run pytest tests/ -v
```
Expected: 전체 PASS

- [ ] **복합 증상 시나리오 테스트**

채팅 UI에서 아래 입력:
```
7살 진돗개가 3일째 구토하고 설사하고 무기력해요
```
Expected:
- `parallel_analysis` 노드 실행 (로그에서 3개 서브에이전트 확인)
- VFS에 `symptom_analysis.json`, `breed_risk.json`, `urgency_score.json` 생성
- 종합 리포트에 긴급도 + 진돗개 취약 질환 + 대처법 포함

- [ ] **대화 지속성(메모리) 확인**

같은 `thread_id`로 두 번째 대화:
```
오늘은 밥을 조금 먹었어요
```
Expected: 에이전트가 이전 대화(진돗개, 구토+설사) 맥락을 기억하고 응답

- [ ] **VFS 파일 내용 확인**

```bash
cat /tmp/dog_agent_vfs/urgency_score.json
```
Expected:
```json
{"type": "urgency_judge", "level": "high", "label": "즉시 병원"}
```
