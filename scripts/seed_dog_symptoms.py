"""
dog-symptoms ES 인덱스에 강아지 증상 샘플 데이터를 적재한다.
실행: uv run python scripts/seed_dog_symptoms.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

from app.agents.tools._es_common import get_es_client_bm25, BM25_INDEX_NAME, CONTENT_FIELD

INDEX_SETTINGS = {
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
    "mappings": {
        "properties": {
            CONTENT_FIELD: {"type": "text", "analyzer": "standard"},
            "metadata": {
                "type": "object",
                "properties": {
                    "symptom": {"type": "keyword"},
                    "urgency": {"type": "keyword"},
                    "source": {"type": "keyword"},
                },
            },
        }
    },
}

SAMPLE_DOCS = [
    {CONTENT_FIELD: "강아지 구토: 위장염, 췌장염, 이물질 섭취가 원인입니다. 하루 3회 이상이거나 혈액이 섞인 구토, 12시간 이상 지속 시 즉시 진찰이 필요합니다.", "metadata": {"symptom": "구토", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 식욕부진: 24~48시간 이상 밥을 먹지 않으면 수의사 진찰을 권장합니다. 스트레스, 위장 질환, 신부전, 간 질환이 원인일 수 있습니다.", "metadata": {"symptom": "식욕부진", "urgency": "low", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 기침: 켄넬코프, 심장 질환, 기관허탈이 원인입니다. 소형견(포메라니안, 치와와)은 기관허탈에 취약합니다. 혈액이 섞이면 즉시 응급 진찰이 필요합니다.", "metadata": {"symptom": "기침", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 설사: 세균성·바이러스성 장염, 기생충, 음식 알레르기가 원인입니다. 혈변이나 탈수 증상이 있으면 즉시 진찰이 필요합니다.", "metadata": {"symptom": "설사", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 무기력증: 빈혈, 감염병, 심장 질환의 신호일 수 있습니다. 구토·식욕부진·창백한 잇몸과 함께 나타나면 즉시 병원이 필요합니다.", "metadata": {"symptom": "무기력증", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 경련: 뇌전증, 독소 섭취, 저혈당의 신호입니다. 5분 이상 지속되거나 하루 2회 이상 반복되면 응급 상황입니다. 즉시 동물병원 응급실로 이동하세요.", "metadata": {"symptom": "경련", "urgency": "high", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 절뚝거림: 슬개골 탈구, 관절염, 골절이 원인입니다. 소형견(푸들, 포메라니안, 말티즈)은 슬개골 탈구 발생률이 높습니다.", "metadata": {"symptom": "절뚝거림", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 피부 가려움증: 알레르기성 피부염, 아토피, 외부 기생충이 원인입니다. 탈모·발적·딱지가 함께 나타나면 수의사 진찰이 필요합니다.", "metadata": {"symptom": "가려움증", "urgency": "low", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 음수량 증가: 당뇨병, 쿠싱증후군, 신부전, 자궁축농증의 신호일 수 있습니다. 중년~노령 암컷에서 복부 팽만이 함께 나타나면 즉시 진찰이 필요합니다.", "metadata": {"symptom": "음수량증가", "urgency": "medium", "source": "dog-symptom-guide"}},
    {CONTENT_FIELD: "강아지 호흡 곤란: 잇몸이 파란색/흰색이거나 입을 벌리고 숨쉬면 심장 질환, 폐렴, 기흉의 응급 신호입니다. 즉시 동물병원 응급실로 이동하세요.", "metadata": {"symptom": "호흡곤란", "urgency": "high", "source": "dog-symptom-guide"}},
]


def seed():
    es = get_es_client_bm25()
    if not es.indices.exists(index=BM25_INDEX_NAME):
        es.indices.create(index=BM25_INDEX_NAME, body=INDEX_SETTINGS)
        print(f"인덱스 생성: {BM25_INDEX_NAME}")
    else:
        print(f"인덱스 이미 존재: {BM25_INDEX_NAME}")
    for i, doc in enumerate(SAMPLE_DOCS):
        es.index(index=BM25_INDEX_NAME, id=str(i + 1), document=doc)
        print(f"  [{i+1}/{len(SAMPLE_DOCS)}] {doc['metadata']['symptom']} 적재")
    es.indices.refresh(index=BM25_INDEX_NAME)
    count = es.count(index=BM25_INDEX_NAME)["count"]
    print(f"\n완료: 총 {count}개 문서")


if __name__ == "__main__":
    seed()
