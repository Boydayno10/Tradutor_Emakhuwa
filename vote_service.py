import re
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from supabase_client_strict import VARIANTS_TABLE_NAME, get_client


VOTES_TABLE = "emakua_variant_votes"


def _normalize(text: str) -> str:
    value = (text or "").strip().lower()
    value = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def _tokenize_words(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", raw) if t.strip()]


def _extract_pairs(source_text: str, translated_text: str) -> List[Tuple[str, str]]:
    src_tokens = _tokenize_words(source_text)

    # For single-word translations the app may show variants separated by comma.
    tgt_primary = (translated_text or "").split(",")[0].strip()
    tgt_tokens = _tokenize_words(tgt_primary)

    if not src_tokens or not tgt_tokens:
        return []

    pair_count = min(len(src_tokens), len(tgt_tokens))
    if pair_count <= 0:
        return []

    pairs: List[Tuple[str, str]] = []
    for idx in range(pair_count):
        pt = src_tokens[idx].strip()
        em = tgt_tokens[idx].strip()
        if not pt or not em:
            continue
        pairs.append((pt, em))
    return pairs


def _upsert_variant_if_missing(pt: str, macua: str) -> None:
    client = get_client()
    norm_pt = _normalize(pt)
    norm_macua = _normalize(macua)
    if not norm_pt or not norm_macua:
        return

    rows = (
        client.table(VARIANTS_TABLE_NAME)
        .select("id")
        .eq("normalized_pt", norm_pt)
        .eq("normalized_macua", norm_macua)
        .limit(1)
        .execute()
    )
    data = getattr(rows, "data", None) or []
    if data:
        return

    client.table(VARIANTS_TABLE_NAME).insert(
        {
            "pt": pt,
            "macua": macua,
            "normalized_pt": norm_pt,
            "normalized_macua": norm_macua,
            "priority": 1000,
        }
    ).execute()


def _recompute_variant_feedback(normalized_pt: str, normalized_macua: str) -> Dict[str, int]:
    client = get_client()
    rows = (
        client.table(VOTES_TABLE)
        .select("vote")
        .eq("normalized_pt", normalized_pt)
        .eq("normalized_macua", normalized_macua)
        .execute()
    )
    votes = getattr(rows, "data", None) or []

    up = 0
    down = 0
    for row in votes:
        val = int((row or {}).get("vote") or 0)
        if val > 0:
            up += 1
        elif val < 0:
            down += 1

    score = up - down
    (
        client.table(VARIANTS_TABLE_NAME)
        .update(
            {
                "votes_up": up,
                "votes_down": down,
                "vote_score": score,
                "last_feedback_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .eq("normalized_pt", normalized_pt)
        .eq("normalized_macua", normalized_macua)
        .execute()
    )
    return {"votes_up": up, "votes_down": down, "vote_score": score}


def register_translation_vote(
    user_id: str,
    source_text: str,
    translated_text: str,
    vote: int,
) -> Dict[str, Any]:
    if vote not in (-1, 1):
        raise RuntimeError("Campo 'vote' deve ser -1 ou 1")

    pairs = _extract_pairs(source_text, translated_text)
    if not pairs:
        raise RuntimeError("Nao foi possivel extrair pares de palavras para votar")

    client = get_client()
    updated: List[Dict[str, Any]] = []

    for pt, macua in pairs:
        _upsert_variant_if_missing(pt, macua)

        norm_pt = _normalize(pt)
        norm_macua = _normalize(macua)
        if not norm_pt or not norm_macua:
            continue

        client.table(VOTES_TABLE).upsert(
            {
                "user_id": user_id,
                "source_pt": pt,
                "target_macua": macua,
                "normalized_pt": norm_pt,
                "normalized_macua": norm_macua,
                "vote": vote,
            },
            on_conflict="user_id,normalized_pt,normalized_macua",
        ).execute()

        stats = _recompute_variant_feedback(norm_pt, norm_macua)
        updated.append(
            {
                "pt": pt,
                "macua": macua,
                "normalized_pt": norm_pt,
                "normalized_macua": norm_macua,
                **stats,
            }
        )

    if not updated:
        raise RuntimeError("Nenhuma variante valida foi atualizada")

    return {
        "saved": len(updated),
        "vote": vote,
        "updated": updated,
    }
