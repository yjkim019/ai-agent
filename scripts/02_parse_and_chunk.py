"""PDF 파싱 + 토큰 기반 청킹 스크립트.

PyMuPDF(fitz)로 페이지별 텍스트를 추출하고,
tiktoken으로 토큰 기반 청킹(500토큰, 50 오버랩)을 수행합니다.

출력: data/chunks/pet_chunks.json

사용법:
    python scripts/02_parse_and_chunk.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from scripts.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DIR,
    CHUNKS_FILE,
    ENCODING_NAME,
    PDF_DIR,
)


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """PDF에서 페이지별 텍스트를 추출합니다."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "source": pdf_path.name,
                "page": page_num + 1,
                "text": text,
            })
    doc.close()
    return pages


def chunk_text_by_tokens(
    text: str,
    encoding: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """토큰 기반으로 텍스트를 청킹합니다."""
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size - chunk_overlap
    return chunks


def main():
    pdf_dir = ROOT_DIR / PDF_DIR
    chunks_dir = ROOT_DIR / CHUNKS_DIR
    chunks_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️  {pdf_dir}/ 에 PDF가 없습니다. 먼저 PDF를 배치하세요.")
        sys.exit(1)

    print("=" * 60)
    print("PDF 파싱 + 토큰 기반 청킹")
    print(f"  청크 크기: {CHUNK_SIZE} 토큰, 오버랩: {CHUNK_OVERLAP} 토큰")
    print("=" * 60)

    encoding = tiktoken.get_encoding(ENCODING_NAME)
    all_chunks: list[dict] = []

    for pdf_path in pdfs:
        print(f"\n📄 {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        print(f"  → {len(pages)} 페이지 추출")

        pdf_chunk_count = 0
        for page_info in pages:
            text = page_info["text"]
            chunks = chunk_text_by_tokens(text, encoding)
            for idx, chunk_text in enumerate(chunks):
                all_chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": page_info["source"],
                        "page": page_info["page"],
                        "chunk_index": idx,
                    },
                })
                pdf_chunk_count += 1
        print(f"  → {pdf_chunk_count} 청크 생성")

    output_path = ROOT_DIR / CHUNKS_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 총 {len(all_chunks)}개 청크 → {output_path}")
    print(f"다음 단계: python scripts/03_generate_embeddings.py")


if __name__ == "__main__":
    main()
