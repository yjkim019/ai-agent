"""강아지 증상 검색 에이전트 (BM25 + kNN 하이브리드)"""
from __future__ import annotations

from typing import TypedDict

from elasticsearch import Elasticsearch
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from opik import track

from app.agents.tools._es_common import BM25_INDEX_NAME, INDEX_NAME, CONTENT_FIELD

# ---------------------------------------------------------------------------
# Elasticsearch 연결 설정
# ---------------------------------------------------------------------------

_ES_URL = "https://elasticsearch-edu.didim365.app"
_ES_USER = "elastic"
_ES_PASSWORD = "FJl79PA7mMIJajxB1OHgdLEe"
_TOP_K = 5
_EMBEDDING_MODEL = "text-embedding-3-small"
_EMBEDDING_DIM = 1536


def _get_es_client() -> Elasticsearch:
    return Elasticsearch(
        _ES_URL,
        basic_auth=(_ES_USER, _ES_PASSWORD),
        verify_certs=False,
    )


@track(name="get_query_vector")
def _get_query_vector(query: str) -> list[float] | None:
    """OpenAI 임베딩으로 쿼리 벡터를 생성합니다."""
    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(model=_EMBEDDING_MODEL, input=[query])
        return resp.data[0].embedding
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 검색 단계별 함수 (Opik 트레이스용)
# ---------------------------------------------------------------------------


@track(name="bm25_search")
def _bm25_search(es: Elasticsearch, query: str) -> dict[str, dict]:
    """BM25 키워드 검색 결과를 {doc_id: {hit, ranks}} 형태로 반환합니다."""
    resp = es.search(
        index=BM25_INDEX_NAME,
        body={
            "query": {"match": {CONTENT_FIELD: {"query": query, "operator": "or"}}},
            "size": _TOP_K * 2,
        },
    )
    hits_map: dict[str, dict] = {}
    for rank, hit in enumerate(resp["hits"]["hits"], 1):
        doc_id = hit["_id"]
        hits_map[doc_id] = {"hit": hit, "ranks": [rank]}
    return hits_map


@track(name="knn_search")
def _knn_search(es: Elasticsearch, query_vector: list[float]) -> dict[str, dict]:
    """kNN 벡터 검색 결과를 {doc_id: {hit, ranks}} 형태로 반환합니다."""
    resp = es.search(
        index=INDEX_NAME,
        body={
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": _TOP_K * 2,
                "num_candidates": 100,
            },
            "size": _TOP_K * 2,
        },
    )
    hits_map: dict[str, dict] = {}
    for rank, hit in enumerate(resp["hits"]["hits"], 1):
        hits_map[hit["_id"]] = {"hit": hit, "ranks": [rank]}
    return hits_map


@track(name="rrf_rerank")
def _rrf_rerank(hits_map: dict[str, dict], top_k: int = _TOP_K) -> list[dict]:
    """RRF(Reciprocal Rank Fusion)로 BM25+kNN 결과를 재정렬합니다."""
    _RRF_K = 60

    def _score(ranks: list[int]) -> float:
        return sum(1.0 / (_RRF_K + r) for r in ranks)

    scored = sorted(hits_map.values(), key=lambda v: _score(v["ranks"]), reverse=True)
    return [v["hit"] for v in scored[:top_k]]


# ---------------------------------------------------------------------------
# LangGraph 검색 StateGraph
# ---------------------------------------------------------------------------


class _SearchState(TypedDict):
    query: str
    result: str


def _search_node(state: _SearchState) -> _SearchState:
    """BM25 + kNN 하이브리드 검색 + RRF 재정렬."""
    query = state["query"]
    try:
        es = _get_es_client()

        # 1) BM25 검색
        hits_map = _bm25_search(es, query)

        # 2) kNN 벡터 검색 후 병합
        query_vector = _get_query_vector(query)
        if query_vector:
            knn_hits = _knn_search(es, query_vector)
            for doc_id, entry in knn_hits.items():
                if doc_id in hits_map:
                    hits_map[doc_id]["ranks"].extend(entry["ranks"])
                else:
                    hits_map[doc_id] = entry

        if not hits_map:
            return {"query": query, "result": f"'{query}'에 대한 관련 정보를 찾을 수 없습니다."}

        # 3) RRF 재정렬
        hits = _rrf_rerank(hits_map)

        results: list[str] = []
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            content = source.get(CONTENT_FIELD, "")
            metadata = source.get("metadata", {})
            pdf_source = metadata.get("source", "")
            page = metadata.get("page", "")
            header = f"[{i}]" + (f" 출처: {pdf_source}" if pdf_source else "") + (f" p.{page}" if page else "")
            snippet = content[:400].replace("\n", " ")
            results.append(f"{header}\n{snippet}")

        return {"query": query, "result": "\n\n".join(results)}
    except Exception as e:
        return {"query": query, "result": f"검색 중 오류가 발생했습니다: {e}"}


def _build_search_graph():
    builder = StateGraph(_SearchState)
    builder.add_node("search", _search_node)
    builder.set_entry_point("search")
    builder.add_edge("search", END)
    return builder.compile()


_search_graph = _build_search_graph()


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
def search_symptoms(search_query: str) -> str:
    """강아지 증상 및 질환 관련 의료 정보를 검색합니다.
    구토, 식욕부진, 기침, 절뚝거림 등 증상과 의심 질환 정보 검색에 사용하세요."""
    result = _search_graph.invoke({"query": search_query})
    return result["result"]
