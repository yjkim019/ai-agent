"""임베딩 벡터 생성 스크립트.

OpenAI text-embedding-3-small 모델로 각 청크의 임베딩 벡터를 생성합니다.

입력: data/chunks/pet_chunks.json
출력: data/chunks/pet_chunks_with_vectors.json

사용법:
    python scripts/03_generate_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from openai import OpenAI

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from scripts.config import CHUNKS_FILE, CHUNKS_WITH_VECTORS_FILE, EMBEDDING_MODEL

# 배치 크기 (OpenAI 임베딩 API 제한 고려)
BATCH_SIZE = 100


def generate_embeddings(chunks: list[dict], client: OpenAI) -> list[dict]:
    """청크 리스트에 content_vector 필드를 추가합니다."""
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["content"] for c in batch]

        print(f"  임베딩 생성 중: {i + 1}~{min(i + BATCH_SIZE, total)} / {total}")
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)

        for j, item in enumerate(response.data):
            chunks[i + j]["content_vector"] = item.embedding

        # rate limit 방지
        if i + BATCH_SIZE < total:
            time.sleep(0.5)

    return chunks


def main():
    chunks_path = ROOT_DIR / CHUNKS_FILE
    if not chunks_path.exists():
        print(f"⚠️  {chunks_path} 파일이 없습니다. 먼저 02_parse_and_chunk.py를 실행하세요.")
        sys.exit(1)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("=" * 60)
    print(f"임베딩 벡터 생성 (모델: {EMBEDDING_MODEL})")
    print(f"  청크 수: {len(chunks)}")
    print("=" * 60)

    client = OpenAI()  # OPENAI_API_KEY 환경변수 자동 인식
    chunks = generate_embeddings(chunks, client)

    output_path = ROOT_DIR / CHUNKS_WITH_VECTORS_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"\n✅ 임베딩 완료 → {output_path}")
    print(f"다음 단계: python scripts/04_index_to_es.py --recreate")


if __name__ == "__main__":
    main()
