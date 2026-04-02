"""반려견 프로필 장기 메모리 서비스 (SQLite 기반).

thread_id별로 품종·나이·체중·진단 이력을 저장/조회합니다.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional


_DEFAULT_DB_PATH = "pet_memory.db"

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pet_profiles (
    thread_id   TEXT PRIMARY KEY,
    breed       TEXT,
    age         TEXT,
    weight      TEXT,
    history     TEXT DEFAULT '[]',
    updated_at  TEXT DEFAULT (datetime('now'))
);
"""


class PetMemoryService:
    """thread_id 기반 반려견 프로필 영속 저장소."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        self.db_path = str(Path(db_path))
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.commit()

    def save_profile(
        self,
        thread_id: str,
        breed: Optional[str] = None,
        age: Optional[str] = None,
        weight: Optional[str] = None,
    ) -> None:
        """프로필을 저장하거나 업데이트합니다. None 값은 기존 값을 유지합니다."""
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT breed, age, weight FROM pet_profiles WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

            if existing:
                new_breed  = breed  if breed  is not None else existing[0]
                new_age    = age    if age    is not None else existing[1]
                new_weight = weight if weight is not None else existing[2]
                conn.execute(
                    """UPDATE pet_profiles
                       SET breed=?, age=?, weight=?, updated_at=datetime('now')
                       WHERE thread_id=?""",
                    (new_breed, new_age, new_weight, thread_id),
                )
            else:
                conn.execute(
                    "INSERT INTO pet_profiles (thread_id, breed, age, weight) VALUES (?,?,?,?)",
                    (thread_id, breed, age, weight),
                )
            conn.commit()

    def add_diagnosis(self, thread_id: str, diagnosis: str) -> None:
        """진단 이력에 항목을 추가합니다."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT history FROM pet_profiles WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

            if row is None:
                history: list[str] = [diagnosis]
                conn.execute(
                    "INSERT INTO pet_profiles (thread_id, history) VALUES (?,?)",
                    (thread_id, json.dumps(history, ensure_ascii=False)),
                )
            else:
                history = json.loads(row[0] or "[]")
                history.append(diagnosis)
                conn.execute(
                    "UPDATE pet_profiles SET history=?, updated_at=datetime('now') WHERE thread_id=?",
                    (json.dumps(history, ensure_ascii=False), thread_id),
                )
            conn.commit()

    def get_profile(self, thread_id: str) -> dict | None:
        """저장된 프로필을 반환합니다. 없으면 None."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT breed, age, weight, history FROM pet_profiles WHERE thread_id = ?",
                (thread_id,),
            ).fetchone()

        if row is None:
            return None
        return {
            "breed":   row[0],
            "age":     row[1],
            "weight":  row[2],
            "history": json.loads(row[3] or "[]"),
        }

    def delete_profile(self, thread_id: str) -> None:
        """프로필을 삭제합니다."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM pet_profiles WHERE thread_id = ?", (thread_id,))
            conn.commit()
