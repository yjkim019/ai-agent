def test_es_index_names():
    from app.agents.tools._es_common import INDEX_NAME, BM25_INDEX_NAME
    assert INDEX_NAME == "dog-knowledge"
    assert BM25_INDEX_NAME == "dog-symptoms"


def test_search_symptoms_tool_name():
    from app.agents.search_agent import search_symptoms
    assert search_symptoms.name == "search_symptoms"
