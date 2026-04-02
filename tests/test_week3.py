"""3주차: LangGraph StateGraph + PDF RAG 테스트"""


def test_parse_chunk_handles_generate_report_node():
    from app.services.agent_service import AgentService
    from langchain_core.messages import AIMessage

    service = AgentService.__new__(AgentService)
    chunk = {
        "generate_report": {
            "messages": [AIMessage(content="**긴급도**: 가정 관찰 가능")]
        }
    }
    events = service._parse_chunk(chunk)
    assert len(events) >= 1
    assert "긴급도" in events[0]


def test_dog_agent_uses_state_graph():
    from app.agents.symptom_pipeline import build_symptom_graph
    graph = build_symptom_graph()
    nodes = list(graph.get_graph().nodes.keys())
    assert "collect_symptoms" in nodes
    assert "ask_follow_up" in nodes
    assert "search_rag" in nodes
    assert "generate_report" in nodes


def test_symptom_pipeline_graph_importable():
    from app.agents.symptom_pipeline import build_symptom_graph
    graph = build_symptom_graph()
    assert graph is not None


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
