"""
Simple in-memory prompt history.

WARNING: This is in-memory state. It resets on restart and is NOT shared
across uvicorn workers. For real conversation memory, use Redis or a DB.
This is the learning version — the embryo of conversation memory for agents.
"""
from collections import deque
from datetime import datetime
from typing import List, Dict


class PromptHistory:
    def __init__(self, max_size: int = 10):
        self._history = deque(maxlen=max_size)

    def add(self, message: str, model: str) -> None:
        self._history.append({
            "message": message,
            "model": model,
            "timestamp": datetime.now().isoformat(),
        })

    def get_all(self) -> List[Dict]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()


# Singleton
prompt_history = PromptHistory(max_size=10)