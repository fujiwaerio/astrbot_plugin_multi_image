"""Bounded local task metadata, without prompts, images, endpoints or raw IDs."""

import hashlib
import re
import sqlite3
import time
import uuid
from contextlib import closing
from pathlib import Path
from typing import ClassVar
from weakref import WeakValueDictionary

STATES = {
    "generating": "请求生成中",
    "preparing_images": "准备图片中",
    "completed": "处理完成（平台投递未确认）",
    "failed": "失败",
    "uncertain": "结果不确定，请核对上游任务或账单",
    "interrupted": "图片准备或发送流程中断",
}


def owner_key(platform: str, group: str, user: str) -> str:
    """Hash the caller and scope so raw platform identifiers are never stored.

    Args:
        platform: Platform identifier.
        group: Group ID or an empty string for private messages.
        user: Caller ID.

    Returns:
        A scope-bound digest for filtering records.
    """
    return hashlib.sha256(repr((platform, group, user)).encode()).hexdigest()


class TaskHistory:
    """Persist a small metadata journal with atomic SQLite transactions."""

    _active: ClassVar[WeakValueDictionary] = WeakValueDictionary()

    def __init__(self, path: Path, keep_days: int = 7):
        self.path = path
        self.run_id = uuid.uuid4().hex
        self.keep_days = max(1, min(365, int(keep_days)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, owner TEXT NOT NULL, channel TEXT NOT NULL, model TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, elapsed REAL NOT NULL DEFAULT 0, images INTEGER NOT NULL DEFAULT 0, run_id TEXT NOT NULL)"
            )
            # A previous instance cannot prove whether an unfinished request succeeded.
            live = tuple(self._active)
            placeholders = ",".join("?" for _ in live) or "''"
            db.execute(
                "UPDATE tasks SET state=CASE WHEN state='generating' THEN 'uncertain' ELSE 'interrupted' END WHERE state IN ('generating', 'preparing_images') AND run_id NOT IN ("
                + placeholders
                + ")",
                live,
            )
            self._prune(db)
        self._active[self.run_id] = self

    def close(self):
        """Mark this instance's unfinished records conservatively on unload."""
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE tasks SET state=CASE WHEN state='generating' THEN 'uncertain' ELSE 'interrupted' END WHERE state IN ('generating','preparing_images') AND run_id=?",
                (self.run_id,),
            )
        self._active.pop(self.run_id, None)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        return db

    def _prune(self, db):
        db.execute(
            "DELETE FROM tasks WHERE created < ?",
            (time.time() - self.keep_days * 86400,),
        )
        db.execute(
            "DELETE FROM tasks WHERE state NOT IN ('generating','preparing_images') AND id NOT IN (SELECT id FROM tasks ORDER BY created DESC, id DESC LIMIT 500)"
        )

    def start(self, owner: str, channel: str, model: str) -> str:
        """Record only safe metadata before any generation submission.

        Args:
            owner: Hashed caller and conversation scope.
            channel: Known provider name.
            model: Configured model identifier, sanitized before storage.

        Returns:
            A random task identifier, independent of upstream request identifiers.
        """
        identifier = uuid.uuid4().hex
        if (
            "://" in model
            or not re.fullmatch(r"[\w./:-]{1,160}", model)
            or re.search(r"(?:sk|ark)-", model, re.IGNORECASE)
        ):
            model = "自定义模型"
        now = time.time()
        with closing(self._connect()) as db, db:
            self._prune(db)
            db.execute(
                "INSERT INTO tasks (id,owner,channel,model,state,created,updated,run_id) VALUES (?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    owner,
                    channel,
                    model,
                    "generating",
                    now,
                    now,
                    self.run_id,
                ),
            )
        return identifier

    def update(self, identifier: str, state: str, elapsed: float, images: int = 0):
        """Update fixed state fields without persisting exception messages.

        Args:
            identifier: Local task ID.
            state: A state from the fixed vocabulary.
            elapsed: Local processing time in seconds.
            images: Number of generated images, when known.
        """
        if state not in STATES:
            raise ValueError("Unknown task state")
        with closing(self._connect()) as db, db:
            db.execute(
                "UPDATE tasks SET state=?,updated=?,elapsed=?,images=? WHERE id=?",
                (state, time.time(), max(0, elapsed), max(0, images), identifier),
            )
            self._prune(db)

    def query(self, owner: str, identifier: str = "") -> list[dict]:
        """Return the caller's last five tasks or one exact task ID.

        Args:
            owner: Hashed caller and conversation scope.
            identifier: Optional exact ID; parameters are always bound.

        Returns:
            Fixed metadata fields for authorized records only.
        """
        with closing(self._connect()) as db, db:
            self._prune(db)
            if identifier:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE owner=? AND id=?", (owner, identifier)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM tasks WHERE owner=? ORDER BY created DESC, id DESC LIMIT 5",
                    (owner,),
                ).fetchall()
            return [dict(row) for row in rows]
