"""3주차: LangGraph StateGraph + PDF RAG 테스트"""


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
