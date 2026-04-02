"""Week 4: VFS 서비스 및 멀티 서브에이전트 테스트"""
import asyncio
import pytest
from app.services.vfs_service import VFSService
from app.services.pet_memory_service import PetMemoryService
from app.agents.parallel_agents import urgency_judge, run_parallel_analysis


def test_vfs_write_and_read(tmp_path):
    """write 후 read 시 동일 데이터를 반환해야 한다."""
    vfs = VFSService(base_dir=str(tmp_path / "vfs"))
    data = {"urgency": "high", "symptoms": ["구토", "식욕부진"]}
    vfs.write("result.json", data)
    loaded = vfs.read("result.json")
    assert loaded == data


def test_vfs_read_missing_returns_none(tmp_path):
    """존재하지 않는 파일을 read 하면 None을 반환해야 한다."""
    vfs = VFSService(base_dir=str(tmp_path / "vfs"))
    assert vfs.read("nonexistent.json") is None


def test_vfs_clear_removes_all_files(tmp_path):
    """clear 후에는 저장된 파일이 없어야 한다."""
    vfs = VFSService(base_dir=str(tmp_path / "vfs"))
    vfs.write("a.json", {"key": "a"})
    vfs.write("b.json", {"key": "b"})
    assert len(vfs.list_files()) == 2
    vfs.clear()
    assert vfs.list_files() == []


def test_vfs_list_files(tmp_path):
    """write한 파일들이 list_files에 포함되어야 한다."""
    vfs = VFSService(base_dir=str(tmp_path / "vfs"))
    vfs.write("symptom_result.json", {"result": "ok"})
    vfs.write("breed_result.json", {"breed": "말티즈"})
    files = vfs.list_files()
    assert "symptom_result.json" in files
    assert "breed_result.json" in files


# ---------------------------------------------------------------------------
# 병렬 서브에이전트 테스트
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_urgency_judge_emergency_keyword():
    """경련 키워드가 포함되면 high 긴급도를 반환해야 한다 (규칙 기반)."""
    result = await urgency_judge("강아지가 갑자기 경련을 일으키고 있어요")
    assert result["type"] == "urgency"
    assert result["urgency"] == "high"
    assert result["method"] == "rule"


@pytest.mark.asyncio
async def test_run_parallel_analysis_structure():
    """run_parallel_analysis 결과에 urgency, symptom_search 키가 있어야 한다."""
    result = await run_parallel_analysis("강아지가 구토를 했어요", breed=None)
    assert "urgency" in result
    assert result["urgency"] in ("high", "medium", "low", "observe")
    # symptom_search는 ES 연결 실패 시 에러 문자열도 허용
    assert "symptom_search" in result


# ---------------------------------------------------------------------------
# 반려견 프로필 메모리 테스트
# ---------------------------------------------------------------------------


def test_pet_memory_save_and_get(tmp_path):
    """save_profile 후 get_profile로 동일 데이터를 반환해야 한다."""
    svc = PetMemoryService(db_path=str(tmp_path / "pet.db"))
    svc.save_profile("thread-1", breed="말티즈", age="3살")
    profile = svc.get_profile("thread-1")
    assert profile is not None
    assert profile["breed"] == "말티즈"
    assert profile["age"] == "3살"


def test_pet_memory_partial_update(tmp_path):
    """None 값은 기존 값을 유지해야 한다."""
    svc = PetMemoryService(db_path=str(tmp_path / "pet.db"))
    svc.save_profile("thread-1", breed="푸들", age="2살")
    svc.save_profile("thread-1", weight="3kg")  # breed/age는 None
    profile = svc.get_profile("thread-1")
    assert profile["breed"] == "푸들"
    assert profile["age"] == "2살"
    assert profile["weight"] == "3kg"


def test_pet_memory_diagnosis_history(tmp_path):
    """add_diagnosis 호출 후 history에 항목이 추가되어야 한다."""
    svc = PetMemoryService(db_path=str(tmp_path / "pet.db"))
    svc.save_profile("thread-1", breed="진도개")
    svc.add_diagnosis("thread-1", "구토 - 중등도")
    svc.add_diagnosis("thread-1", "식욕부진 - 경증")
    profile = svc.get_profile("thread-1")
    assert "구토 - 중등도" in profile["history"]
    assert "식욕부진 - 경증" in profile["history"]


def test_pet_memory_missing_returns_none(tmp_path):
    """존재하지 않는 thread_id는 None을 반환해야 한다."""
    svc = PetMemoryService(db_path=str(tmp_path / "pet.db"))
    assert svc.get_profile("nonexistent") is None
