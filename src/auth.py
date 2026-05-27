"""Shared user authentication and role management.

Stores users in a local JSON file with salted SHA-256 password hashes.

Pre-created accounts (seeded on first launch):
  admin    / admin123    — system administrator
  operator / operator123 — platform operator
  user     / user123     — regular user

New registrations default to role="user" (普通用户).  An admin must promote
a user to "operator" or "admin" via the approval endpoint.
"""
import hashlib
import json
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from src.config.settings import ROOT

DEFAULT_DATA_PATH = ROOT / "data" / "users.json"


@dataclass
class User:
    username: str
    password_hash: str
    salt: str
    role: str = "user"       # "admin" | "operator" | "user"
    approved: bool = True     # new registrations are auto-approved as "user"


class AuthStore:
    """Thread-safe JSON-backed user store."""

    def __init__(self, path: Optional[Path] = None):
        self._path = Path(path) if path else DEFAULT_DATA_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._users: Dict[str, User] = {}
        self._load()
        self._seed()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                self._users = {k: User(**v) for k, v in raw.items()}
            except Exception:
                self._users = {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k: v.__dict__ for k, v in self._users.items()}, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._path)

    # -- seed accounts ------------------------------------------------------

    def _seed(self) -> None:
        changed = False
        for username, role in [("admin", "admin"), ("operator", "operator"), ("user", "user")]:
            if username not in self._users:
                self._users[username] = self._create_user(username, username + "123", role)
                changed = True
        if changed:
            self._save()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _hash(password: str, salt: Optional[str] = None) -> tuple:
        salt = salt or secrets.token_hex(16)
        h = hashlib.sha256((salt + password).encode("utf-8")).hexdigest()
        return h, salt

    def _create_user(self, username: str, password: str, role: str) -> User:
        pw_hash, salt = self._hash(password)
        return User(username=username, password_hash=pw_hash, salt=salt, role=role, approved=True)

    # -- public API ---------------------------------------------------------

    def authenticate(self, username: str, password: str) -> Optional[User]:
        with self._lock:
            user = self._users.get(username)
        if user is None:
            return None
        h, _ = self._hash(password, user.salt)
        if h == user.password_hash:
            return user
        return None

    def register(self, username: str, password: str) -> Optional[User]:
        with self._lock:
            if username in self._users:
                return None
            user = self._create_user(username, password, role="user")
            self._users[username] = user
            self._save()
        return user

    def get_user(self, username: str) -> Optional[User]:
        with self._lock:
            return self._users.get(username)

    def list_users(self) -> List[dict]:
        with self._lock:
            return [
                {"username": u.username, "role": u.role, "approved": u.approved}
                for u in self._users.values()
            ]

    def promote_user(self, username: str, new_role: str) -> bool:
        if new_role not in ("admin", "operator", "user"):
            return False
        with self._lock:
            user = self._users.get(username)
            if user is None:
                return False
            user.role = new_role
            self._save()
        return True

    def delete_user(self, username: str) -> bool:
        with self._lock:
            if username not in self._users:
                return False
            del self._users[username]
            self._save()
        return True


# Global singleton
_auth_store: Optional[AuthStore] = None
_lock = threading.Lock()


def get_auth_store(path: Optional[Path] = None) -> AuthStore:
    global _auth_store
    if _auth_store is None:
        with _lock:
            if _auth_store is None:
                _auth_store = AuthStore(path=path)
    return _auth_store
