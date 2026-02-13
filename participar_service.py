import random
from collections import defaultdict
from typing import Any, Dict, List

from supabase_client_strict import get_client


PUBLIC_CONFIG_TABLE = "project_public_config"
PRACTICE_BANK_TABLE = "project_practice_bank"


def _word_count(text: str) -> int:
    return len([t for t in (text or "").split() if t.strip()])


def get_participar_config(max_per_topic: int = 3) -> Dict[str, Any]:
    client = get_client()

    privacy_url = ""
    try:
        cfg_resp = (
            client.table(PUBLIC_CONFIG_TABLE)
            .select("key,value")
            .eq("key", "privacy_policy_url")
            .limit(1)
            .execute()
        )
        cfg_rows = getattr(cfg_resp, "data", None) or []
        if cfg_rows:
            privacy_url = str((cfg_rows[0] or {}).get("value") or "").strip()
    except Exception:
        privacy_url = ""

    questions_by_topic: Dict[str, List[str]] = defaultdict(list)
    try:
        q_resp = (
            client.table(PRACTICE_BANK_TABLE)
            .select("topic,pt_text,max_words")
            .eq("active", True)
            .limit(500)
            .execute()
        )
        rows = getattr(q_resp, "data", None) or []
    except Exception:
        rows = []

    by_topic_candidates: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        topic = str((row or {}).get("topic") or "Geral").strip() or "Geral"
        pt_text = str((row or {}).get("pt_text") or "").strip()
        max_words = int((row or {}).get("max_words") or 4)
        if not pt_text:
            continue
        wc = _word_count(pt_text)
        if wc < 2 or wc > max_words or wc > 4:
            continue
        by_topic_candidates[topic].append(pt_text)

    for topic, items in by_topic_candidates.items():
        uniq = list(dict.fromkeys(items))
        random.shuffle(uniq)
        questions_by_topic[topic] = uniq[: max(1, max_per_topic)]

    # Fallback minimal set if DB has no active rows.
    if not questions_by_topic:
        questions_by_topic["Geral"] = [
            "A lingua local vive",
            "Nosso povo aprende",
            "Casa bonita",
        ]

    return {
        "privacy_policy_url": privacy_url,
        "practice_topics": [
            {"topic": topic, "questions": questions}
            for topic, questions in questions_by_topic.items()
        ],
    }

