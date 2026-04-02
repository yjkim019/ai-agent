"""Elasticsearch 인덱스 생성 + 벌크 적재 스크립트.

pet-knowledge (벡터 인덱스)와 pet-collection (BM25 인덱스)에 청크를 적재합니다.

입력: data/chunks/pet_chunks_with_vectors.json

사용법:
    python scripts/04_index_to_es.py              # 기존 인덱스에 추가
    python scripts/04_index_to_es.py --recreate   # 인덱스 삭제 후 재생성
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from scripts.config import (
    BM25_INDEX,
    CHUNKS_WITH_VECTORS_FILE,
    EMBEDDING_DIM,
    VECTOR_INDEX,
)

# ---------------------------------------------------------------------------
# ES 접속 — 환경변수 또는 기본값
# ---------------------------------------------------------------------------
import os

ES_URL = os.getenv("ES_PET_URL", "https://elasticsearch-edu.didim365.app")
ES_USER = os.getenv("ES_PET_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PET_PASSWORD", "FJl79PA7mMIJajxB1OHgdLEe")


def get_es() -> Elasticsearch:
    return Elasticsearch(ES_URL, basic_auth=(ES_USER, ES_PASSWORD), verify_certs=False)


# ---------------------------------------------------------------------------
# 인덱스 매핑 정의
# ---------------------------------------------------------------------------

def _detect_analyzer(es: Elasticsearch) -> str:
    """nori 플러그인 설치 여부를 확인하고 분석기명을 반환합니다."""
    try:
        plugins = es.cat.plugins(format="json")
        plugin_names = [p.get("component", "") for p in plugins]
        if any("nori" in name for name in plugin_names):
            return "nori"
    except Exception:
        pass
    return "standard"


VECTOR_INDEX_SETTINGS = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIM,
                "index": True,
                "similarity": "cosine",
            },
            "metadata": {
                "properties": {
                    "source": {"type": "keyword"},
                    "page": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                }
            },
        }
    }
}


def _bm25_index_settings(analyzer: str) -> dict:
    settings: dict = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": analyzer},
                "metadata": {
                    "properties": {
                        "source": {"type": "keyword"},
                        "page": {"type": "integer"},
                        "chunk_index": {"type": "integer"},
                    }
                },
            }
        }
    }
    if analyzer == "nori":
        settings["settings"] = {
            "analysis": {
                "analyzer": {
                    "nori": {
                        "type": "custom",
                        "tokenizer": "nori_tokenizer",
                    }
                }
            }
        }
    return settings


# ---------------------------------------------------------------------------
# 인덱스 생성/삭제
# ---------------------------------------------------------------------------


def recreate_index(es: Elasticsearch, index_name: str, body: dict):
    if es.indices.exists(index=index_name):
        print(f"  인덱스 삭제: {index_name}")
        es.indices.delete(index=index_name)
    print(f"  인덱스 생성: {index_name}")
    es.indices.create(index=index_name, body=body)


def ensure_index(es: Elasticsearch, index_name: str, body: dict):
    if not es.indices.exists(index=index_name):
        print(f"  인덱스 생성: {index_name}")
        es.indices.create(index=index_name, body=body)
    else:
        print(f"  인덱스 존재: {index_name}")


# ---------------------------------------------------------------------------
# 벌크 적재
# ---------------------------------------------------------------------------


def bulk_index_vector(es: Elasticsearch, chunks: list[dict]):
    """pet-knowledge 벡터 인덱스에 적재합니다."""
    actions = []
    for i, chunk in enumerate(chunks):
        doc = {
            "content": chunk["content"],
            "content_vector": chunk["content_vector"],
            "metadata": chunk.get("metadata", {}),
        }
        actions.append({"_index": VECTOR_INDEX, "_id": str(i), "_source": doc})

    success, errors = bulk(es, actions, raise_on_error=False)
    print(f"  [{VECTOR_INDEX}] 적재 완료: {success}건 성공, {len(errors) if isinstance(errors, list) else 0}건 오류")


def bulk_index_bm25(es: Elasticsearch, chunks: list[dict]):
    """pet-collection BM25 인덱스에 적재합니다."""
    actions = []
    for i, chunk in enumerate(chunks):
        doc = {
            "content": chunk["content"],
            "metadata": chunk.get("metadata", {}),
        }
        actions.append({"_index": BM25_INDEX, "_id": str(i), "_source": doc})

    success, errors = bulk(es, actions, raise_on_error=False)
    print(f"  [{BM25_INDEX}] 적재 완료: {success}건 성공, {len(errors) if isinstance(errors, list) else 0}건 오류")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Elasticsearch 인덱스 생성 + 벌크 적재")
    parser.add_argument("--recreate", action="store_true", help="기존 인덱스를 삭제하고 재생성")
    args = parser.parse_args()

    chunks_path = ROOT_DIR / CHUNKS_WITH_VECTORS_FILE
    if not chunks_path.exists():
        print(f"⚠️  {chunks_path} 파일이 없습니다. 먼저 03_generate_embeddings.py를 실행하세요.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("=" * 60)
    print("Elasticsearch 인덱스 적재")
    print(f"  ES URL: {ES_URL}")
    print(f"  벡터 인덱스: {VECTOR_INDEX}")
    print(f"  BM25 인덱스: {BM25_INDEX}")
    print(f"  청크 수: {len(chunks)}")
    print(f"  재생성 모드: {args.recreate}")
    print("=" * 60)

    es = get_es()

    # 분석기 감지
    analyzer = _detect_analyzer(es)
    print(f"\n텍스트 분석기: {analyzer}")

    bm25_settings = _bm25_index_settings(analyzer)

    # 인덱스 설정
    if args.recreate:
        print("\n[인덱스 재생성]")
        recreate_index(es, VECTOR_INDEX, VECTOR_INDEX_SETTINGS)
        recreate_index(es, BM25_INDEX, bm25_settings)
    else:
        print("\n[인덱스 확인]")
        ensure_index(es, VECTOR_INDEX, VECTOR_INDEX_SETTINGS)
        ensure_index(es, BM25_INDEX, bm25_settings)

    # 벌크 적재
    print("\n[벌크 적재]")
    bulk_index_vector(es, chunks)
    bulk_index_bm25(es, chunks)

    # 검증
    es.indices.refresh(index=VECTOR_INDEX)
    es.indices.refresh(index=BM25_INDEX)
    vec_count = es.count(index=VECTOR_INDEX)["count"]
    bm25_count = es.count(index=BM25_INDEX)["count"]
    print(f"\n✅ 적재 완료")
    print(f"  {VECTOR_INDEX}: {vec_count}건")
    print(f"  {BM25_INDEX}: {bm25_count}건")


if __name__ == "__main__":
    main()
