# Elasticsearch configuration and common utilities

# Vector / Hybrid 검색용 인덱스
INDEX_NAME = "dog-knowledge"
# BM25 키워드 검색용 인덱스
BM25_INDEX_NAME = "dog-symptoms"

# 기본 콘텐츠 필드명
CONTENT_FIELD = "content"

# Elasticsearch 연결 정보
_ES_URL = "https://elasticsearch-edu.didim365.app"
_ES_USER = "elastic"
_ES_PASSWORD = "FJl79PA7mMIJajxB1OHgdLEe"


def get_es_client_bm25():
    """BM25 검색용 Elasticsearch 클라이언트를 반환합니다."""
    from elasticsearch import Elasticsearch
    return Elasticsearch(
        _ES_URL,
        basic_auth=(_ES_USER, _ES_PASSWORD),
        verify_certs=False,
    )
