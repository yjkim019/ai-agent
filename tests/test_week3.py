"""3주차: LangGraph ReAct 에이전트 + PDF RAG 테스트"""


def test_parse_chunk_model_step():
    """model 스텝의 ChatResponse tool_call을 done 이벤트로 변환해야 한다."""
    import json
    from app.services.agent_service import AgentService
    from unittest.mock import MagicMock

    service = AgentService.__new__(AgentService)

    msg = MagicMock()
    msg.tool_calls = [
        {
            "name": "ChatResponse",
            "args": {
                "message_id": "test-id",
                "content": "**긴급도**: 가정 관찰 가능",
                "metadata": {"urgency": "observe"},
            },
        }
    ]
    chunk = {"model": {"messages": [msg]}}
    events = service._parse_chunk(chunk)
    assert len(events) >= 1
    payload = json.loads(events[0])
    assert payload["step"] == "done"
    assert "긴급도" in payload["content"]


def test_dog_agent_importable():
    """dog_agent 모듈이 정상적으로 임포트되어야 한다."""
    from app.agents.dog_agent import create_dog_agent
    assert callable(create_dog_agent)


def test_dog_agent_tools_registered():
    """dog_agent에 search_symptoms, get_pet_breed_info, find_nearby_vet 도구가 등록되어야 한다."""
    from app.agents.search_agent import search_symptoms
    from app.agents.tools import get_pet_breed_info, find_nearby_vet
    assert search_symptoms.name == "search_symptoms"
    assert get_pet_breed_info.name == "get_pet_breed_info"
    assert find_nearby_vet.name == "find_nearby_vet"


def test_script_config_dog_domain():
    import importlib.util, os
    spec = importlib.util.spec_from_file_location(
        "scripts_config",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "config.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert "dog" in mod.VECTOR_INDEX_NAME
    assert "dog" in mod.BM25_INDEX_NAME
