import re
import unicodedata
from datetime import timedelta
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

from supabase_client_strict import VARIANTS_TABLE_NAME, get_client


VOTES_TABLE = "emakua_variant_votes"
COOLDOWN_TABLE = "emakua_vote_cooldowns"

# Progressivo por texto normalizado (mesma palavra/frase)
# 1o voto: 21m, 2o: 45m, 3o: 70m, 4o: 120m, depois continua crescendo.
_COOLDOWN_MINUTES_STEPS = [21, 45, 70, 120, 180, 240, 360, 480, 720, 1440]


class VoteCooldownError(RuntimeError):
    def __init__(self, message: str, retry_after_seconds: int, next_allowed_at: str = ""):
        super().__init__(message)
        self.retry_after_seconds = max(1, int(retry_after_seconds))
        self.next_allowed_at = str(next_allowed_at or "")


def _normalize(text: str) -> str:
    value = (text or "").strip().lower()
    value = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def _normalize_source_text(text: str) -> str:
    raw = (text or "").strip().lower()
    raw = re.sub(r"\s+", " ", raw)
    raw = unicodedata.normalize("NFD", raw)
    return "".join(ch for ch in raw if unicodedata.category(ch) != "Mn")


def _tokenize_words(text: str) -> List[str]:
    raw = (text or "").strip()
    if not raw:
        return []
    return [t for t in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", raw) if t.strip()]


def _cooldown_minutes_for_vote_count(vote_count_after: int) -> int:
    if vote_count_after <= 0:
        return _COOLDOWN_MINUTES_STEPS[0]
    idx = vote_count_after - 1
    if idx < len(_COOLDOWN_MINUTES_STEPS):
        return _COOLDOWN_MINUTES_STEPS[idx]

    # Continua crescendo de forma controlada apos a tabela base.
    last = _COOLDOWN_MINUTES_STEPS[-1]
    extra_steps = idx - (len(_COOLDOWN_MINUTES_STEPS) - 1)
    return last + (extra_steps * 180)


def _format_wait_delta(seconds_remaining: int) -> str:
    secs = max(1, int(seconds_remaining))
    hours = secs // 3600
    minutes = (secs % 3600) // 60
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{secs}s"


def _check_and_advance_cooldown(user_id: str, source_text: str) -> Dict[str, Any]:
    client = get_client()
    normalized_source = _normalize_source_text(source_text)
    if not normalized_source:
        raise RuntimeError("Texto de origem invalido para votacao")

    rows_resp = (
        client.table(COOLDOWN_TABLE)
        .select("id,vote_count,next_allowed_at")
        .eq("user_id", user_id)
        .eq("normalized_source_text", normalized_source)
        .limit(1)
        .execute()
    )
    rows = getattr(rows_resp, "data", None) or []

    now = datetime.now(timezone.utc)
    current_count = 0
    row_id = None
    next_allowed_at_raw = None
    if rows:
        row = rows[0] or {}
        row_id = str(row.get("id") or "").strip() or None
        current_count = int(row.get("vote_count") or 0)
        next_allowed_at_raw = str(row.get("next_allowed_at") or "").strip()

    if next_allowed_at_raw:
        try:
            next_allowed = datetime.fromisoformat(next_allowed_at_raw.replace("Z", "+00:00"))
        except Exception:
            next_allowed = None
        if next_allowed is not None and next_allowed > now:
            remaining = int((next_allowed - now).total_seconds())
            message = (
                "Para evitar abusos, você só pode votar novamente "
                f"nesta mesma tradução daqui a {_format_wait_delta(remaining)}."
            )
            raise VoteCooldownError(
                message=message,
                retry_after_seconds=remaining,
                next_allowed_at=next_allowed.isoformat(),
            )

    new_count = current_count + 1
    cooldown_minutes = _cooldown_minutes_for_vote_count(new_count)
    next_allowed_at = now + timedelta(minutes=cooldown_minutes)

    payload = {
        "user_id": user_id,
        "source_text": source_text,
        "normalized_source_text": normalized_source,
        "vote_count": new_count,
        "last_voted_at": now.isoformat(),
        "next_allowed_at": next_allowed_at.isoformat(),
    }

    if row_id:
        (
            client.table(COOLDOWN_TABLE)
            .update(payload)
            .eq("id", row_id)
            .execute()
        )
    else:
        client.table(COOLDOWN_TABLE).insert(payload).execute()

    return {
        "vote_count": new_count,
        "next_allowed_at": next_allowed_at.isoformat(),
        "cooldown_minutes": cooldown_minutes,
    }


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

    cooldown_info = _check_and_advance_cooldown(user_id, source_text)

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
        "cooldown": cooldown_info,
    }
