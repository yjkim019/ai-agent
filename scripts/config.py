# 청킹 설정
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ENCODING_NAME = "cl100k_base"

# 임베딩 모델
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Elasticsearch 인덱스 (dog 도메인)
VECTOR_INDEX = "dog-knowledge"    # kNN 벡터 검색용
BM25_INDEX = "dog-symptoms"       # BM25 키워드 검색용
VECTOR_INDEX_NAME = "dog-knowledge"
BM25_INDEX_NAME = "dog-symptoms"

# 파일 경로
PDF_DIR = "data/pdfs"
CHUNKS_DIR = "data/chunks"
CHUNKS_FILE = "data/chunks/dog_chunks.json"
CHUNKS_WITH_VECTORS_FILE = "data/chunks/dog_chunks_with_vectors.json"
CHUNKS_OUTPUT = "data/chunks/dog_chunks.json"
CHUNKS_WITH_VECTORS = "data/chunks/dog_chunks_with_vectors.json"
