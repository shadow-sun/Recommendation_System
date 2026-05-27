"""User authentication — re-exports from the shared auth module."""
from src.auth import AuthStore, User, get_auth_store

__all__ = ["AuthStore", "User", "get_auth_store"]
