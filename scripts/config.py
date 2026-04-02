VECTOR_INDEX_NAME = "dog-knowledge"       # Vector kNN 검색용
BM25_INDEX_NAME = "dog-symptoms"          # BM25 키워드 검색용
CHUNK_SIZE = 500                           # 토큰 단위 청킹 크기
CHUNK_OVERLAP = 50                         # 청킹 오버랩
PDF_DIR = "data/pdfs"                     # PDF 저장 디렉터리
CHUNKS_OUTPUT = "data/chunks/dog_chunks.json"
CHUNKS_WITH_VECTORS = "data/chunks/dog_chunks_with_vectors.json"
