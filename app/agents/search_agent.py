"""강아지 증상 검색 에이전트 (BM25 + kNN 하이브리드)"""
from __future__ import annotations

from typing import TypedDict

from elasticsearch import Elasticsearch
from langchain.tools import tool
from langgraph.graph import StateGraph, END

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
# LangGraph 검색 StateGraph
# ---------------------------------------------------------------------------


class _SearchState(TypedDict):
    query: str
    result: str


_RRF_K = 60  # RRF 상수 (일반적으로 60 사용)


def _rrf_score(ranks: list[int]) -> float:
    """Reciprocal Rank Fusion 점수 계산."""
    return sum(1.0 / (_RRF_K + r) for r in ranks)


def _search_node(state: _SearchState) -> _SearchState:
    """BM25 + kNN 하이브리드 검색 + RRF 재정렬."""
    query = state["query"]
    try:
        es = _get_es_client()
        # doc_id → {"hit": hit, "ranks": [rank_from_bm25, rank_from_knn]}
        hits_map: dict[str, dict] = {}

        # 1) BM25 검색
        bm25_resp = es.search(
            index=BM25_INDEX_NAME,
            body={
                "query": {"match": {CONTENT_FIELD: {"query": query, "operator": "or"}}},
                "size": _TOP_K * 2,
            },
        )
        for rank, hit in enumerate(bm25_resp["hits"]["hits"], 1):
            doc_id = hit["_id"]
            hits_map[doc_id] = {"hit": hit, "ranks": [rank]}

        # 2) kNN 벡터 검색
        query_vector = _get_query_vector(query)
        if query_vector:
            knn_resp = es.search(
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
            for rank, hit in enumerate(knn_resp["hits"]["hits"], 1):
                doc_id = hit["_id"]
                if doc_id in hits_map:
                    hits_map[doc_id]["ranks"].append(rank)
                else:
                    hits_map[doc_id] = {"hit": hit, "ranks": [rank]}

        if not hits_map:
            return {"query": query, "result": f"'{query}'에 대한 관련 정보를 찾을 수 없습니다."}

        # 3) RRF 점수로 재정렬
        scored = sorted(
            hits_map.values(),
            key=lambda v: _rrf_score(v["ranks"]),
            reverse=True,
        )[:_TOP_K]
        hits = [v["hit"] for v in scored]
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
