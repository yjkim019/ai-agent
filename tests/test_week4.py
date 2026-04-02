"""Week 4: VFS 서비스 및 멀티 서브에이전트 테스트"""
import pytest
from app.services.vfs_service import VFSService


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
