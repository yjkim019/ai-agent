def test_es_index_names():
    from app.agents.tools._es_common import INDEX_NAME, BM25_INDEX_NAME
    assert INDEX_NAME == "dog-knowledge"
    assert BM25_INDEX_NAME == "dog-symptoms"


def test_search_symptoms_tool_name():
    from app.agents.search_agent import search_symptoms
    assert search_symptoms.name == "search_symptoms"


def test_dog_symptom_prompt_keywords():
    from app.agents.prompts import DOG_SYMPTOM_SYSTEM_PROMPT
    assert "강아지" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "긴급도" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "search_symptoms" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "get_pet_breed_info" in DOG_SYMPTOM_SYSTEM_PROMPT
    assert "find_nearby_vet" in DOG_SYMPTOM_SYSTEM_PROMPT


def test_create_dog_agent_importable():
    from app.agents.dog_agent import create_dog_agent
    assert callable(create_dog_agent)


def test_main_app_title_contains_dog():
    from app.main import app
    assert "강아지" in app.title
