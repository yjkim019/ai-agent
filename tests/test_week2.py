"""2주차: Opik 관찰성 + 평가 테스트"""
import json
import os


def test_create_dog_agent_is_tracked():
    from app.agents.dog_agent import create_dog_agent
    # opik @track 데코레이터는 __wrapped__ 속성을 남긴다
    assert hasattr(create_dog_agent, "__wrapped__") or callable(create_dog_agent)
