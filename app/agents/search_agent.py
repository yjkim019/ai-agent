"""강아지 증상 검색 에이전트 (BM25 기반)"""
from __future__ import annotations

from typing import Any, TypedDict

from elasticsearch import Elasticsearch
from langchain.tools import tool
from langgraph.graph import StateGraph, END

from app.agents.tools._es_common import BM25_INDEX_NAME, CONTENT_FIELD

# ---------------------------------------------------------------------------
# Elasticsearch 연결 설정
# ---------------------------------------------------------------------------

_ES_URL = "https://elasticsearch-edu.didim365.app"
_ES_USER = "elastic"
_ES_PASSWORD = "FJl79PA7mMIJajxB1OHgdLEe"
_TOP_K = 5


def _get_es_client() -> Elasticsearch:
    return Elasticsearch(
        _ES_URL,
        basic_auth=(_ES_USER, _ES_PASSWORD),
        verify_certs=False,
    )


# ---------------------------------------------------------------------------
# LangGraph 검색 StateGraph
# ---------------------------------------------------------------------------


class _SearchState(TypedDict):
    query: str
    result: str


def _search_node(state: _SearchState) -> _SearchState:
    """BM25 검색을 수행하고 결과를 반환합니다."""
    query = state["query"]
    try:
        es = _get_es_client()
        resp = es.search(
            index=BM25_INDEX_NAME,
            body={
                "query": {
                    "match": {
                        CONTENT_FIELD: {
                            "query": query,
                            "operator": "or",
                        }
                    }
                },
                "size": _TOP_K,
            },
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return {"query": query, "result": f"'{query}'에 대한 관련 정보를 찾을 수 없습니다."}

        results: list[str] = []
        for i, hit in enumerate(hits, 1):
            source = hit.get("_source", {})
            content = source.get(CONTENT_FIELD, "")
            metadata = source.get("metadata", {})
            symptom = metadata.get("symptom", "")
            urgency = metadata.get("urgency", "")
            header = f"[{i}] 증상: {symptom}" + (f" (긴급도: {urgency})" if urgency else "")
            snippet = content[:500].replace("\n", " ")
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
