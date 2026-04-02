# 3주차: LangGraph StateGraph + PDF RAG 고도화 플랜

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 단계적 질문으로 증상을 좁혀가는 StateGraph 에이전트 + 수의학 PDF 기반 하이브리드 검색(BM25 + kNN + RRF)으로 종합 리포트 품질 향상

**Architecture:** `create_agent` → `StateGraph`로 전환. 노드: `collect_symptoms` → `ask_follow_up`(최대 3회) → `search_rag`(BM25+kNN+RRF) → `generate_report`. PDF 파이프라인 스크립트는 기존 `scripts/` 재활용하되 인덱스명·설정을 강아지 도메인으로 교체.

**Tech Stack:** LangGraph `StateGraph`, Elasticsearch hybrid search + RRF, PyMuPDF, tiktoken, Cohere ReRank

**사전 조건:** 2주차 플랜 완료, `dog-symptoms` 인덱스 존재, 수의학 PDF 파일 수집 완료

---

## 파일 구조

| 파일 | 변경 종류 | 내용 |
|---|---|---|
| `scripts/config.py` | 수정 | 인덱스명·경로를 강아지 도메인으로 교체 |
| `scripts/02_parse_and_chunk.py` | 수정 | 청킹 설정 확인 (500토큰, 50 오버랩 유지) |
| `scripts/04_index_to_es.py` | 수정 | `dog-knowledge` 인덱스에 벡터 적재 |
| `app/agents/symptom_pipeline.py` | 신규 | StateGraph 기반 증상 분석 파이프라인 |
| `app/agents/dog_agent.py` | 수정 | `create_dog_agent`를 StateGraph 기반으로 교체 |
| `app/services/agent_service.py` | 수정 | StateGraph 스트리밍 파싱 업데이트 |
| `tests/test_week3.py` | 신규 | 단위 테스트 |

---

### Task 1: PDF 파이프라인 설정 업데이트

**Files:**
- Modify: `scripts/config.py`
- Test: `tests/test_week3.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week3.py
def test_script_config_dog_domain():
    import importlib.util, os, sys
    spec = importlib.util.spec_from_file_location(
        "scripts_config",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "dog" in mod.VECTOR_INDEX_NAME
    assert "dog" in mod.BM25_INDEX_NAME
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week3.py::test_script_config_dog_domain -v
```
Expected: FAIL

- [ ] **Step 3: `scripts/config.py` 인덱스명 교체**

현재 파일을 읽고 아래 값으로 교체:

```python
# scripts/config.py
VECTOR_INDEX_NAME = "dog-knowledge"      # Vector kNN 검색용
BM25_INDEX_NAME = "dog-symptoms"         # BM25 키워드 검색용
CHUNK_SIZE = 500                          # 토큰 단위 청킹 크기
CHUNK_OVERLAP = 50                        # 청킹 오버랩
PDF_DIR = "data/pdfs"                    # PDF 저장 디렉터리
CHUNKS_OUTPUT = "data/chunks/dog_chunks.json"
CHUNKS_WITH_VECTORS = "data/chunks/dog_chunks_with_vectors.json"
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week3.py::test_script_config_dog_domain -v
```
Expected: PASS

- [ ] **Step 5: PDF 수집 및 적재 실행**

수의학 PDF를 `data/pdfs/` 폴더에 배치한 뒤 순서대로 실행:

```bash
uv run python scripts/02_parse_and_chunk.py
uv run python scripts/03_generate_embeddings.py
uv run python scripts/04_index_to_es.py
```
Expected: `dog-knowledge` 인덱스에 벡터 문서 적재 완료

- [ ] **Step 6: 커밋**

```bash
git add scripts/config.py tests/test_week3.py
git commit -m "feat: update PDF pipeline config for dog domain"
```

---

### Task 2: `symptom_pipeline.py` StateGraph 구현

**Files:**
- Create: `app/agents/symptom_pipeline.py`
- Test: `tests/test_week3.py`

StateGraph 흐름:
```
START → collect_symptoms → route
  route → ask_follow_up (question_count < 3 AND 정보 부족)
  route → search_rag     (question_count >= 3 OR 정보 충분)
ask_follow_up → END      (사용자 응답 대기)
search_rag → generate_report → END
```

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week3.py 에 추가
def test_symptom_pipeline_graph_importable():
    from app.agents.symptom_pipeline import build_symptom_graph
    graph = build_symptom_graph()
    assert graph is not None
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week3.py::test_symptom_pipeline_graph_importable -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: `symptom_pipeline.py` 생성**

```python
# app/agents/symptom_pipeline.py
"""StateGraph 기반 강아지 증상 분석 파이프라인.

흐름:
  collect_symptoms → (정보 충분?) → search_rag → generate_report
                  ↘ (부족) → ask_follow_up → END (사용자 응답 대기)
"""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.agents.prompts import DOG_SYMPTOM_SYSTEM_PROMPT
from app.agents.search_agent import search_symptoms
from app.agents.tools import get_pet_breed_info


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SymptomState(TypedDict):
    messages: Annotated[list, add_messages]
    question_count: int   # 에이전트가 추가 질문한 횟수


# ---------------------------------------------------------------------------
# 유틸리티
# ---------------------------------------------------------------------------


def _get_llm() -> ChatOpenAI:
    from app.core.config import settings
    from pydantic import SecretStr
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=SecretStr(settings.OPENAI_API_KEY),
    )


def _has_enough_info(state: SymptomState) -> bool:
    """HumanMessage가 2개 이상이면 충분한 정보로 판단한다."""
    human_count = sum(1 for m in state["messages"] if isinstance(m, HumanMessage))
    return human_count >= 2


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def collect_symptoms(state: SymptomState) -> dict:
    """첫 번째 노드: 상태 초기화만 수행 (메시지는 이미 state에 있음)."""
    return {}


def ask_follow_up(state: SymptomState) -> dict:
    """LLM이 추가 질문을 생성한다."""
    follow_up_questions = [
        "언제부터 그런 증상이 있었나요?",
        "밥이나 물은 먹고 있나요?",
        "다른 이상한 행동도 보이나요? (기침, 설사, 무기력 등)",
    ]
    count = state["question_count"]
    if count < len(follow_up_questions):
        question = follow_up_questions[count]
    else:
        llm = _get_llm()
        resp = llm.invoke(
            [SystemMessage(content="강아지 증상 추가 정보를 얻기 위한 짧은 질문 하나만 하세요.")]
            + state["messages"]
        )
        question = resp.content

    return {
        "messages": [AIMessage(content=question)],
        "question_count": state["question_count"] + 1,
    }


def search_rag(state: SymptomState) -> dict:
    """BM25+Vector+RRF 하이브리드 검색으로 관련 질환 정보를 가져온다."""
    user_content = " ".join(
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    )
    search_result = search_symptoms.invoke({"search_query": user_content})
    # 검색 결과를 SystemMessage로 삽입해 generate_report 노드에서 활용
    return {
        "messages": [SystemMessage(content=f"[검색 결과]\n{search_result}")]
    }


def generate_report(state: SymptomState) -> dict:
    """수집된 증상 + 검색 결과를 바탕으로 종합 리포트를 작성한다."""
    llm = _get_llm()
    report_instruction = SystemMessage(
        content=(
            DOG_SYMPTOM_SYSTEM_PROMPT
            + "\n\n위 대화와 검색 결과를 바탕으로 종합 리포트를 작성하세요. "
            "반드시 긴급도·의심 질환·가정 대처법·병원 방문 시 전달사항을 포함하세요."
        )
    )
    response = llm.invoke([report_instruction] + state["messages"])
    return {"messages": [AIMessage(content=response.content)]}


# ---------------------------------------------------------------------------
# 라우팅
# ---------------------------------------------------------------------------


def route_after_collect(state: SymptomState) -> Literal["ask_follow_up", "search_rag"]:
    """충분한 정보가 있거나 질문 횟수가 3회에 도달하면 검색으로 진행."""
    if state["question_count"] >= 3 or _has_enough_info(state):
        return "search_rag"
    return "ask_follow_up"


# ---------------------------------------------------------------------------
# 그래프 구성
# ---------------------------------------------------------------------------


def build_symptom_graph(checkpointer=None):
    builder = StateGraph(SymptomState)

    builder.add_node("collect_symptoms", collect_symptoms)
    builder.add_node("ask_follow_up", ask_follow_up)
    builder.add_node("search_rag", search_rag)
    builder.add_node("generate_report", generate_report)

    builder.add_edge(START, "collect_symptoms")
    builder.add_conditional_edges("collect_symptoms", route_after_collect)
    builder.add_edge("ask_follow_up", END)          # 사용자 응답 대기
    builder.add_edge("search_rag", "generate_report")
    builder.add_edge("generate_report", END)

    return builder.compile(checkpointer=checkpointer)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week3.py::test_symptom_pipeline_graph_importable -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/symptom_pipeline.py tests/test_week3.py
git commit -m "feat: create symptom_pipeline StateGraph with follow-up questions"
```

---

### Task 3: `dog_agent.py` StateGraph로 전환

**Files:**
- Modify: `app/agents/dog_agent.py`
- Test: `tests/test_week3.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week3.py 에 추가
def test_dog_agent_uses_state_graph():
    from app.agents.symptom_pipeline import build_symptom_graph
    graph = build_symptom_graph()
    # get_graph()로 노드 목록 확인
    nodes = list(graph.get_graph().nodes.keys())
    assert "collect_symptoms" in nodes
    assert "ask_follow_up" in nodes
    assert "search_rag" in nodes
    assert "generate_report" in nodes
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week3.py::test_dog_agent_uses_state_graph -v
```
Expected: PASS (graph 구조 확인) — 아직 dog_agent.py 미연결이면 FAIL

- [ ] **Step 3: `dog_agent.py` StateGraph 전환**

기존 `create_agent` 기반 코드를 아래로 교체:

```python
# app/agents/dog_agent.py
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
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week3.py -v
```
Expected: PASS

- [ ] **Step 5: 커밋**

```bash
git add app/agents/dog_agent.py tests/test_week3.py
git commit -m "feat: replace create_agent with StateGraph in dog_agent.py"
```

---

### Task 4: `agent_service.py` StateGraph 스트리밍 파싱 업데이트

> `create_agent` → `StateGraph` 전환으로 청크 키가 `model`/`tools`에서 `collect_symptoms` 등으로 바뀐다.

**Files:**
- Modify: `app/services/agent_service.py:114-149`
- Test: `tests/test_week3.py`

- [ ] **Step 1: 테스트 작성**

```python
# tests/test_week3.py 에 추가
def test_parse_chunk_handles_generate_report_node():
    from app.services.agent_service import AgentService
    from langchain_core.messages import AIMessage

    service = AgentService.__new__(AgentService)
    chunk = {
        "generate_report": {
            "messages": [AIMessage(content="**긴급도**: 가정 관찰 가능")]
        }
    }
    events = service._parse_chunk(chunk)
    assert len(events) >= 1
    assert "긴급도" in events[0]
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
uv run pytest tests/test_week3.py::test_parse_chunk_handles_generate_report_node -v
```
Expected: FAIL — `generate_report` 키를 파싱하지 못함

- [ ] **Step 3: `agent_service.py` `_parse_chunk` 메서드 교체 (114~149번째 줄)**

```python
def _parse_chunk(self, chunk: dict):
    """StateGraph 스트림 청크를 SSE 이벤트 문자열 리스트로 변환한다."""
    import json as _json
    events: list[str] = []
    for step, event in chunk.items():
        if not event:
            continue
        messages = event.get("messages", []) if isinstance(event, dict) else []
        if not messages:
            continue
        message = messages[-1]  # 노드의 마지막 메시지 사용

        # 기존 create_agent 호환: model 노드의 ChatResponse tool call
        if step == "model":
            tool_calls = getattr(message, "tool_calls", [])
            if tool_calls:
                first_tool = tool_calls[0]
                if first_tool.get("name") == "ChatResponse":
                    args = first_tool.get("args", {})
                    events.append(self._done_event(
                        content=args.get("content", ""),
                        metadata=self._handle_metadata(args.get("metadata")),
                        message_id=args.get("message_id"),
                    ))
                else:
                    events.append(_json.dumps({
                        "step": "model",
                        "tool_calls": [tc["name"] for tc in tool_calls],
                    }))

        # StateGraph: tools 노드
        elif step == "tools":
            events.append(
                f'{{"step": "tools", "name": {_json.dumps(message.name)}, "content": {message.content}}}'
            )

        # StateGraph: ask_follow_up 노드 — 중간 질문 반환
        elif step == "ask_follow_up":
            events.append(self._done_event(content=message.content))

        # StateGraph: generate_report 노드 — 최종 리포트 반환
        elif step == "generate_report":
            events.append(self._done_event(content=message.content))

    return events
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
uv run pytest tests/test_week3.py::test_parse_chunk_handles_generate_report_node -v
```
Expected: PASS

- [ ] **Step 5: 전체 테스트 실행**

```bash
uv run pytest tests/test_week1.py tests/test_week2.py tests/test_week3.py -v
```
Expected: 전체 PASS

- [ ] **Step 6: 커밋**

```bash
git add app/services/agent_service.py tests/test_week3.py
git commit -m "feat: update agent_service streaming parser for StateGraph nodes"
```

---

## 3주차 최종 검증

- [ ] **서버 구동 및 단계적 질문 흐름 확인**

```bash
uv run uvicorn app.main:app --reload --port 8000
```

채팅 UI에서 아래 시나리오 테스트:
```
턴 1: "강아지가 구토를 해요"
      → 에이전트: "언제부터 그런 증상이 있었나요?"
턴 2: "어제부터요"
      → 에이전트: "밥이나 물은 먹고 있나요?"
턴 3: "밥을 안 먹어요"
      → 에이전트: 종합 리포트 생성 (긴급도·의심 질환·대처법)
```

- [ ] **하이브리드 검색 동작 확인**

PDF 적재 후 검색 결과에 PDF 출처(source)가 포함되는지 확인:
```bash
uv run python -c "
from dotenv import load_dotenv; load_dotenv()
from app.agents.search_agent import search_symptoms
result = search_symptoms.invoke({'search_query': '구토 췌장염'})
print(result[:300])
"
```
Expected: PDF 출처 정보가 포함된 검색 결과 출력
