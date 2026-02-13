import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_PATH = BASE_DIR / "Arquivos de treinamento" / "processed" / "training_knowledge.json"


@lru_cache(maxsize=1)
def load_training_knowledge() -> Dict[str, Any]:
    if not KNOWLEDGE_PATH.exists():
        return {}
    try:
        raw = KNOWLEDGE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def clear_training_knowledge_cache() -> None:
    load_training_knowledge.cache_clear()

