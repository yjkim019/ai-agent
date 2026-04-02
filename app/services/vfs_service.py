"""VFS(Virtual File System) 서비스 - 서브에이전트 중간 결과 저장용"""
from __future__ import annotations

import json
import os
from pathlib import Path


class VFSService:
    """서브에이전트 중간 결과를 JSON 파일로 저장/조회하는 임시 파일 시스템."""

    def __init__(self, base_dir: str = "/tmp/dog_agent_vfs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, data: dict) -> None:
        """데이터를 JSON 파일로 저장합니다."""
        path = self.base_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def read(self, filename: str) -> dict | None:
        """저장된 JSON 파일을 읽어 반환합니다. 파일이 없으면 None 반환."""
        path = self.base_dir / filename
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_files(self) -> list[str]:
        """저장된 파일 목록을 반환합니다."""
        return [f.name for f in self.base_dir.iterdir() if f.is_file()]

    def clear(self) -> None:
        """저장된 모든 파일을 삭제합니다."""
        for f in self.base_dir.iterdir():
            if f.is_file():
                f.unlink()
