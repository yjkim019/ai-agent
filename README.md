# 강아지 증상 분석 챗봇 — 현황 분석 및 개선 방향

작성일: 2026-04-03

---

## 1. 프로젝트 개요

반려견 보호자가 강아지의 이상 증상을 입력하면 AI가 의심 질환·긴급도·가정 대처법을 분석해 제공하는 의료 보조 챗봇.
수의사 진료를 대체하지 않고, **정보 접근성과 초기 대응 판단**을 보조하는 것이 핵심 가치다.

| 레이어 | 기술 |
|--------|------|
| LLM | OpenAI GPT (`OPENAI_MODEL` 환경변수로 주입) |
| 에이전트 프레임워크 | LangGraph + LangChain `create_agent()` (ReAct) |
| 검색 엔진 | Elasticsearch (BM25 + kNN 하이브리드 + RRF) |
| 임베딩 | OpenAI `text-embedding-3-small` (dim=1536) |
| 서버 | FastAPI + SSE 스트리밍 |
| 대화 이력 | LangGraph `AsyncSqliteSaver` (`checkpoints.db`) |
| 평가 | Opik (LLM-as-a-judge + Heuristic 메트릭) |
| 추적 | Opik Tracer (`track_langgraph` 통합) |

---

## 2. 전체 요청 처리 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│  사용자 입력                                                      │
│  POST /chat  { thread_id, message }                             │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  FastAPI  chat_router                                           │
│  StreamingResponse(media_type="text/event-stream")              │
│  → 최초 이벤트: {"step": "model", "tool_calls": ["Planning"]}   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  AgentService.process_query()                                   │
│                                                                 │
│  1. _init_checkpointer()                                        │
│     └── aiosqlite.connect("checkpoints.db")                     │
│         AsyncSqliteSaver(conn)  → self.checkpointer             │
│                                                                 │
│  2. _create_agent()   ◀─── LangChain create_agent() 호출       │
│     └── create_agent(                                           │
│           model=ChatOpenAI(...),                                │
│           tools=[search_symptoms,                               │
│                  get_pet_breed_info,                            │
│                  find_nearby_vet],                              │
│           system_prompt=DOG_SYMPTOM_SYSTEM_PROMPT,              │
│           response_format=ToolStrategy(ChatResponse),           │
│           checkpointer=self.checkpointer   ◀── SQLite 연결      │
│         )                                                       │
│     └── track_langgraph(agent, opik_tracer)  ← Opik 래핑       │
│                                                                 │
│  3. _patch_pending_tool_calls()  ← 미완료 tool_call 복구        │
│  4. _stream_agent()              ← asyncio.wait 기반 스트리밍   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│  LangGraph ReAct Agent  (create_agent 내부 그래프)              │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  model 노드  (LLM 추론)                                   │  │
│  │                                                          │  │
│  │  DOG_SYMPTOM_SYSTEM_PROMPT 기반으로                      │  │
│  │  아래 두 가지를 판단:                                     │  │
│  │                                                          │  │
│  │  ┌── 유형 A (일반 대화) ──────────────────────────────┐  │  │
│  │  │  조건: 증상 없음 / 정보 부족                        │  │  │
│  │  │  → metadata: {}                                    │  │  │
│  │  │  → content: 자연스러운 텍스트                       │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  ┌── 유형 B (증상 리포트) ────────────────────────────┐  │  │
│  │  │  조건: 이상 증상 1개 이상 + 보조 정보 2개 이상     │  │  │
│  │  │  → metadata: {"urgency": "high|medium|low|observe"}│  │  │
│  │  │  → content: 리포트 형식 마크다운                   │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  [도구 호출 필요 시]  ChatResponse(ToolStrategy) 아닌    │  │
│  │  tool_call 반환 → tools 노드로 라우팅                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                      │                                          │
│          ┌───────────┼────────────────────────────┐            │
│          │           │                            │            │
│          ▼           ▼                            ▼            │
│  ┌──────────┐ ┌──────────────────┐ ┌──────────────────────┐   │
│  │ Tool 1   │ │ Tool 2           │ │ Tool 3               │   │
│  │search_   │ │get_pet_          │ │find_nearby_vet       │   │
│  │symptoms  │ │breed_info        │ │                      │   │
│  │          │ │                  │ │건강보험심사평가원 API │   │
│  │LangGraph │ │하드코딩 딕셔너리 │ │clCd=92 (동물병원)    │   │
│  │서브그래프│ │(16개 품종)       │ │                      │   │
│  │호출 ◀   │ │                  │ │                      │   │
│  └──────────┘ └──────────────────┘ └──────────────────────┘   │
│       │                                                         │
│       │  ToolMessage 반환 → model 노드로 루프 (ReAct)           │
│       │  recursion_limit=15 도달 시 폴백 응답                   │
└───────┼─────────────────────────────────────────────────────────┘
        │
        ▼ (Tool 1만 LangGraph 서브그래프 호출)
┌─────────────────────────────────────────────────────────────────┐
│  search_agent.py — SearchGraph (LangGraph 서브그래프)           │
│                                                                 │
│  1. _get_query_vector(query)                                    │
│     └── OpenAI embeddings.create("text-embedding-3-small")      │
│         → query_vector: list[float] (1536차원)                  │
│                                                                 │
│  2. SearchGraph.invoke(state)                                   │
│                                                                 │
│     START                                                       │
│       ├── bm25_node  ─────────────────────────────┐  (병렬)    │
│       │   ES index: dog-symptoms (BM25)           │            │
│       │   match query, size=10                    │            │
│       │                                           │            │
│       └── knn_node  ──────────────────────────────┤  (병렬)    │
│           ES index: dog-knowledge (kNN)           │            │
│           dense_vector cosine, k=10               │            │
│                                           (fan-in)│            │
│                                                   ▼            │
│                                           rerank_node          │
│                                           RRF(k=60)            │
│                                           score = Σ 1/(60+rank)│
│                                           상위 5개 반환         │
│                                                   │            │
│                                                   ▼            │
│                                           content[:400]        │
│                                           제어문자 제거         │
│                                           (re.sub [\x00-\x1f]) │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│  SSE 이벤트 스트림 (클라이언트로 전송)                           │
│                                                                 │
│  step="model"  → {"step":"model","tool_calls":["search_..."]}  │
│  step="tools"  → {"step":"tools","name":"...","content":"..."}  │
│  step="done"   → {"step":"done","message_id":"...","content":  │
│                   "...","metadata":{"urgency":"..."},...}       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. LangChain create_agent 호출 상세

`dog_agent.py`에서 LangChain의 `create_agent()`로 에이전트를 생성한다.

```python
# app/agents/dog_agent.py

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

agent = create_agent(
    model=model,                             # ChatOpenAI 인스턴스
    tools=[
        search_symptoms,                     # LangGraph 서브그래프 래핑 tool
        get_pet_breed_info,                  # 하드코딩 딕셔너리 tool
        find_nearby_vet,                     # 외부 API tool
    ],
    system_prompt=DOG_SYMPTOM_SYSTEM_PROMPT, # 응답 유형·긴급도·품종 규칙 포함
    response_format=ToolStrategy(ChatResponse),  # 구조화된 최종 응답 강제
    checkpointer=checkpointer,               # AsyncSqliteSaver (멀티턴 이력)
)
```

**ToolStrategy(ChatResponse)의 역할:**
LLM이 도구를 호출하지 않고 최종 응답을 생성할 때, `ChatResponse` 스키마(`message_id`, `content`, `metadata`)를 tool_call 형태로 강제 출력시킨다. `AgentService._parse_chunk()`에서 `first_tool.get("name") == "ChatResponse"` 조건으로 이를 감지해 `step=done` 이벤트를 만든다.

---

## 4. RAG 데이터 파이프라인

### 4-1. 파이프라인 단계

```
data/pdfs/*.pdf          (수동 배치)
        │
        ▼
[02_parse_and_chunk.py]
  라이브러리: PyMuPDF (fitz), tiktoken
  처리:
    - fitz.open() → 페이지별 텍스트 추출 (get_text("text"))
    - tiktoken.get_encoding("cl100k_base")
    - 토큰 기반 슬라이딩 윈도우 청킹
        chunk_size   = 500 토큰
        chunk_overlap =  50 토큰
  출력: data/chunks/dog_chunks.json
    [{"content": "...", "metadata": {"source": "xxx.pdf", "page": 3, "chunk_index": 0}}, ...]
        │
        ▼
[03_generate_embeddings.py]
  라이브러리: openai
  처리:
    - 배치 100개씩 OpenAI API 호출
    - model: text-embedding-3-small
    - 각 청크에 content_vector(1536차원) 필드 추가
    - rate limit 방지: 배치 사이 0.5초 sleep
  출력: data/chunks/dog_chunks_with_vectors.json
        │
        ▼
[04_index_to_es.py]
  라이브러리: elasticsearch (bulk helper)
  처리:
    - nori 플러그인 감지 → 분석기 자동 선택 (nori / standard)
    - 두 인덱스에 동시 적재
  인덱스 A: dog-knowledge   (kNN 벡터 검색용)
    mapping: content(text) + content_vector(dense_vector, cosine, dim=1536)
  인덱스 B: dog-symptoms    (BM25 키워드 검색용)
    mapping: content(text, analyzer=nori|standard)
  옵션: --recreate → 인덱스 삭제 후 재생성
        │
        ▼
[seed_dog_symptoms.py]  ← BM25 인덱스용 시드 데이터 별도 삽입
  (증상·질환 매핑 데이터 직접 삽입)
```

### 4-2. 인덱스 구성 요약

| 인덱스명 | 용도 | 필드 | 분석기 |
|----------|------|------|--------|
| `dog-knowledge` | kNN 벡터 검색 | content, content_vector(1536d), metadata | — |
| `dog-symptoms` | BM25 키워드 검색 | content, metadata | nori 또는 standard |

> **주의:** `tools.py`(레거시)는 `edu-collection` 인덱스를 여전히 참조하고 있으나, 실제 에이전트가 사용하는 `search_symptoms`는 `search_agent.py`의 것으로 `dog-knowledge` / `dog-symptoms`를 사용한다.

### 4-3. 파이프라인 개선 이력

| 문제 | 원인 | 해결 |
|------|------|------|
| RAG 결과에 조류·기생충 정보 혼입 | 무관 PDF 2종 포함 | PDF 제거 후 전체 재파이프라인 |
| OpenAI API 400 오류 | PDF 텍스트 내 제어문자(□■ 등) | `re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", content)` |

---

## 5. Checkpoint (대화 이력 관리)

### 5-1. 구조

```
AgentService._init_checkpointer()
    │
    ├── aiosqlite.connect("checkpoints.db")   ← 비동기 SQLite 연결
    └── AsyncSqliteSaver(conn)                ← LangGraph 체크포인터
            │
            └── create_agent(checkpointer=...)에 주입
                    └── LangGraph 내부에서 자동 관리
                            ├── 매 노드 실행 후 state 저장
                            └── thread_id 기반 이전 대화 복구
```

### 5-2. thread_id 흐름

```
POST /chat  { "thread_id": "3fa85f64-...", "message": "..." }
    │
    ▼
AgentService.process_query(user_messages, thread_id)
    │
    ▼
config = {
    "configurable": {"thread_id": str(thread_id)},
    "recursion_limit": 15
}
    │
    ▼
agent.astream({"messages": [HumanMessage(...)]}, config=config)
    │
    └── AsyncSqliteSaver가 thread_id 키로 checkpoints.db에서
        이전 messages 복원 → 멀티턴 대화 유지
```

### 5-3. 미완료 tool_call 복구 로직

서버 재시작·에러 등으로 tool_call 응답이 없는 상태가 체크포인트에 남으면
다음 요청 시 LangGraph가 오류를 낸다. 이를 방지하기 위한 패치 로직이 있다.

```
process_query() 진입 시
    │
    ▼
_patch_pending_tool_calls(config)
    ├── agent.aget_state(config)             ← 체크포인트에서 현재 state 조회
    ├── _find_pending_tool_calls(messages)   ← ToolMessage 없는 tool_call 탐색
    └── agent.aupdate_state(                 ← 더미 ToolMessage로 패치
            config,
            {"messages": [ToolMessage(
                content="[오류로 인해 실행되지 않았습니다]",
                tool_call_id=tc["id"]
            )]},
            as_node="tools"
        )
```

### 5-4. GraphRecursionError 처리

```
recursion_limit(15) 도달
    │
    ▼
_handle_recursion_fallback(config)
    ├── aget_state() → 현재까지 수집된 messages 조회
    ├── 미완료 tool_calls 더미 ToolMessage 보충
    ├── SystemMessage("지금까지 수집된 정보로 답변하세요.") 추가
    └── model.ainvoke(messages) → LLM 단독 폴백 응답 생성
```

### 5-5. 현재 checkpointer 한계

| 항목 | 현황 | 한계 |
|------|------|------|
| 저장소 | `checkpoints.db` (로컬 파일) | 서버 스케일아웃 불가, 인스턴스 공유 안 됨 |
| thread 목록 조회 | GET /threads가 정적 JSON 읽기 | 실제 대화 목록을 DB에서 조회 못 함 |
| 프로필 지속 | 대화 메시지만 저장 | 품종·나이 등 반려견 정보는 매번 다시 물어봄 |
| 연결 수명 | `AgentService` 인스턴스마다 새 연결 | 평가 시 매 task마다 새 연결 생성 (이벤트 루프 문제 방지용) |

---

## 6. 평가 (Opik 기반)

### 6-1. 평가 흐름

```
run_evaluation.py
    │
    ├── app/data/eval_dataset.json 로드
    ├── opik.Opik().get_or_create_dataset("dog-symptom-eval-v1")
    └── evaluate(
            experiment_name="dog-symptom-agent-week3",
            task=sync_task,
            scoring_metrics=[...],
            task_threads=1
        )
            │
            └── sync_task(item)
                    └── asyncio.run(evaluation_task(item))
                            └── AgentService() 새로 생성  ← 이벤트 루프 충돌 방지
                                    │
                                    └── process_query() 실행 → SSE 파싱
                                            ├── step="model" → tools_used 수집
                                            └── step="done"  → full_response 수집
```

### 6-2. 현재 평가 지표

| 지표 | 유형 | 설명 | 한계 |
|------|------|------|------|
| `answer_relevance` | LLM-as-a-judge (gpt-4.1-mini) | 답변이 증상 질문과 관련 있는가 | 비결정적, 모델 의존 |
| `hallucination` | LLM-as-a-judge (gpt-4.1-mini) | 존재하지 않는 약/치료법 언급 여부 | context 품질에 민감 |
| `urgency_accuracy` | Heuristic (키워드 매칭) | expected_urgency 키워드가 content 텍스트에 포함됐는가 | 표현 방식 변경 시 미감지 |
| `tool_call_accuracy` | Heuristic (집합 포함) | expected_tools ⊆ tools_used 인지 | 과도한 도구 호출 미감지 |

**urgency_accuracy 키워드 맵:**

| urgency | 감지 키워드 |
|---------|------------|
| high | 즉시 병원, 응급, 즉각, 즉시 동물병원 |
| medium | 24시간, 며칠 내, 빠른, 조속히, 신속히 |
| low | 가정 관찰, 천천히, 지켜보, 며칠 내, 천천히 관찰 |
| observe | 가정 관찰, 지켜보 |

---

## 7. 미연결 구성 요소

구현은 완료됐으나 에이전트·API에 연결되지 않은 기능들이다.

| 구성 요소 | 파일 | 기능 | 연결 방법 (제안) |
|-----------|------|------|-----------------|
| `PetMemoryService` | `services/pet_memory_service.py` | thread_id별 품종·나이·체중·진단 이력 SQLite 저장 | 에이전트 도구로 등록 (`save_pet_profile`, `get_pet_profile`) |
| `ConversationService` | `services/conversation_service.py` | 대화 목록 메모리 저장 (재시작 시 초기화) | SQLite/Redis 교체 후 threads API 연결 |
| `VFSService` | `services/vfs_service.py` | `/tmp/dog_agent_vfs/`에 JSON 파일 임시 저장 | 서브에이전트 중간 결과 저장용으로 설계됨 |
| `get_medication_info` | `agents/tools.py` | 식품의약품안전처 e약은요 API 약물 조회 | 에이전트 tools 배열에 추가 검토 |
| `find_nearby_hospitals` | `agents/tools.py` | 건강보험심사평가원 병원 조회 (사람 병원) | `find_nearby_vet`과 통합 또는 제거 |

---

## 8. 개선 방향

### 8-1. 단기 (즉시 가능)

| 항목 | 내용 |
|------|------|
| urgency_accuracy 개선 | content 텍스트가 아닌 `metadata.urgency` 필드를 직접 비교 |
| eval_dataset 확충 | 카테고리(high/medium/low/observe/일반/품종특이)별 3건 이상 추가 → 최소 18건 |
| tools.py 레거시 정리 | `edu-collection` → `dog-symptoms` 인덱스 참조로 통일 |

### 8-2. 중기 (1~2주)

| 항목 | 내용 |
|------|------|
| PetMemoryService 연결 | `thread_id` 기반 반려견 프로필 저장·조회 도구로 에이전트에 추가, 대화 간 품종·나이 기억 |
| 검색 횟수 LangGraph State 강제 | 프롬프트 의존 대신 State에 `tool_call_count` 추가해 6회 초과 시 코드 레벨 차단 |
| threads API 실화 | `checkpoints.db`에서 thread 목록 실시간 조회, 정적 JSON 대체 |

### 8-3. 장기 (구조 개선)

| 항목 | 내용 |
|------|------|
| tools 패키지 구조 정리 | `importlib.util` 동적 import 우회 제거, `tools.py` → `tools/legacy.py` 이동 |
| 품종 데이터 확장 | 하드코딩 16개 → 국내 등록 주요 품종 50종 이상 또는 ES 인덱스에 품종 데이터 통합 |
| ConversationService 영속화 | SQLite 또는 Redis로 전환, 서버 재시작 후에도 대화 목록 유지 |
| tool_call_accuracy F1화 | precision + recall 동시 측정으로 과도한 도구 호출도 감지 |

### 8-4. 보류 (현재 규모에서 과도한 것)

| 항목 | 보류 이유 |
|------|----------|
| Redis 대화 캐시 | SQLite 체크포인터로 충분, 동시 접속 규모 작음 |
| 멀티모달 사진 업로드 | 수의학 이미지 학습 데이터·검증 비용 큼 |
| 실시간 GPS 기반 병원 검색 | 공공 API는 시도 단위까지만 지원, 위치 정보 법적 이슈 |
| 자동 PDF 크롤링 파이프라인 | 저작권·라이선스 검토 선행 필요 |

---

## 9. 파일 맵

```
app/
├── agents/
│   ├── dog_agent.py           # create_agent() 에이전트 팩토리
│   ├── prompts.py             # 시스템 프롬프트 (응답 유형·긴급도·품종 규칙)
│   ├── search_agent.py        # BM25+kNN SearchGraph (LangGraph 서브그래프)
│   ├── tools.py               # 레거시 도구 원본 (edu-collection 참조 포함)
│   └── tools/
│       ├── __init__.py        # tools.py 동적 재수출 (importlib.util 우회)
│       └── _es_common.py      # ES 인덱스명 상수 공유
├── services/
│   ├── agent_service.py       # SSE 스트리밍·체크포인터·Opik 통합
│   ├── pet_memory_service.py  # 반려견 프로필 SQLite 저장 [미연결]
│   ├── conversation_service.py# 대화 목록 메모리 저장 [미연결]
│   ├── vfs_service.py         # 임시 파일 저장 [미연결]
│   └── threads_service.py     # 정적 JSON 읽기 전용
├── api/routes/
│   ├── chat.py                # POST /chat → SSE StreamingResponse
│   └── threads.py             # GET /threads, /threads/{id}, /favorites/questions
└── core/config.py             # pydantic-settings 기반 설정

scripts/
├── 02_parse_and_chunk.py      # PyMuPDF 추출 + tiktoken 청킹 (500t / 50t overlap)
├── 03_generate_embeddings.py  # OpenAI text-embedding-3-small, 배치 100개
├── 04_index_to_es.py          # ES 벌크 적재 (dog-knowledge + dog-symptoms)
├── config.py                  # 파이프라인 상수 (인덱스명, 청크 설정, 경로)
├── seed_dog_symptoms.py       # BM25 인덱스 시드 데이터 삽입
└── run_evaluation.py          # Opik 평가 실행 (4개 메트릭)

data/
├── pdfs/                      # 수의학 PDF 원본
├── chunks/
│   ├── dog_chunks.json                # 파싱+청킹 결과
│   └── dog_chunks_with_vectors.json   # 임베딩 추가 결과
└── eval_dataset.json          # 평가 데이터셋

checkpoints.db                 # LangGraph 대화 이력 (AsyncSqliteSaver)
```
