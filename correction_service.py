from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from supabase_client_strict import VARIANTS_TABLE_NAME, get_client
from translation_pipeline import (
    _build_indexes,
    _levenshtein,
    _normalize_pt,
    load_resources,
    lookup_pt_to_em,
)


def _normalize_macua(text: str) -> str:
    return (text or "").strip().lower()


def _dedupe_pairs(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for item in items:
        pt = str(item.get("pt") or "").strip()
        macua = str(item.get("macua") or "").strip()
        if not pt or not macua:
            continue
        key = (_normalize_pt(pt), _normalize_macua(macua))
        if key in seen:
            continue
        seen.add(key)
        out.append({"pt": pt, "macua": macua})
    return out


def _fetch_rows_by_norm_pt(norm_pt: str) -> List[Dict[str, Any]]:
    client = get_client()
    resp = (
        client
        .table(VARIANTS_TABLE_NAME)
        .select("id,pt,macua,normalized_pt,normalized_macua")
        .eq("normalized_pt", norm_pt)
        .execute()
    )
    return getattr(resp, "data", None) or []


def _fetch_rows_by_fuzzy_norm_pt(norm_pt: str, max_distance: int = 2) -> List[Dict[str, Any]]:
    client = get_client()
    # Pull a bounded set and filter in Python to keep compatibility simple.
    resp = (
        client
        .table(VARIANTS_TABLE_NAME)
        .select("id,pt,macua,normalized_pt,normalized_macua")
        .limit(4000)
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    out: List[Dict[str, Any]] = []
    for row in rows:
        candidate = str((row or {}).get("normalized_pt") or "").strip()
        if not candidate:
            continue
        if candidate == norm_pt:
            out.append(row)
            continue
        if abs(len(candidate) - len(norm_pt)) > max_distance:
            continue
        if _levenshtein(candidate, norm_pt) <= max_distance:
            out.append(row)
    return out


def _pt_norm_distance_limit(norm_pt: str) -> int:
    if len(norm_pt) <= 5:
        return 1
    if len(norm_pt) <= 8:
        return 2
    return 3


def _collect_variants_for_word(word: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, _, _ = _build_indexes(resources)

    lookup = lookup_pt_to_em(word, lexicon_pt, pronoun_pt, spell_vocab_pt)
    resolved_pt = lookup.get("normalized") or _normalize_pt(word)
    principal_pt_raw = (word or "").strip()
    if not principal_pt_raw:
        principal_pt_raw = word

    pairs: List[Dict[str, str]] = []
    for candidate in lookup.get("candidates", []) or []:
        pairs.append({"pt": principal_pt_raw, "macua": str(candidate)})

    dist_limit = _pt_norm_distance_limit(resolved_pt)
    # Include close PT keys from lexicon metadata.
    for pt_norm, macuas in lexicon_pt.items():
        if abs(len(pt_norm) - len(resolved_pt)) > dist_limit:
            continue
        if _levenshtein(pt_norm, resolved_pt) > dist_limit:
            continue
        for macua in macuas:
            pairs.append({"pt": pt_norm, "macua": str(macua)})

    # Include variants saved as independent DB rows.
    db_rows = _fetch_rows_by_norm_pt(resolved_pt)
    if not db_rows:
        db_rows = _fetch_rows_by_fuzzy_norm_pt(resolved_pt, max_distance=dist_limit)
    for row in db_rows:
        pt = str((row or {}).get("pt") or "").strip()
        macua = str((row or {}).get("macua") or "").strip()
        if not pt or not macua:
            continue
        pairs.append({"pt": pt, "macua": macua})

    deduped = _dedupe_pairs(pairs)
    principal = deduped[0] if deduped else {"pt": principal_pt_raw, "macua": ""}
    return principal, deduped


def get_correction_payload(word: str) -> Dict[str, Any]:
    principal, variants = _collect_variants_for_word(word)
    return {
        "input": word,
        "principal": principal,
        "variantes": variants,
    }


def upsert_variants(variantes: List[Dict[str, str]]) -> Dict[str, Any]:
    client = get_client()
    normalized = _dedupe_pairs(variantes)
    if not normalized:
        return {"saved": 0, "variantes": []}

    rows = []
    touched_norm_pts = set()
    for item in normalized:
        pt = item["pt"].strip()
        macua = item["macua"].strip()
        norm_pt = _normalize_pt(pt)
        norm_macua = _normalize_macua(macua)
        touched_norm_pts.add(norm_pt)
        rows.append(
            {
                "pt": pt,
                "macua": macua,
                "normalized_pt": norm_pt,
                "normalized_macua": norm_macua,
            }
        )

    (
        client
        .table(VARIANTS_TABLE_NAME)
        .upsert(
            rows,
            on_conflict="normalized_pt,normalized_macua",
        )
        .execute()
    )

    # Keep only the submitted set for each normalized PT touched in this save.
    for norm_pt in touched_norm_pts:
        allowed = {
            row["normalized_macua"]
            for row in rows
            if row["normalized_pt"] == norm_pt
        }
        existing = _fetch_rows_by_norm_pt(norm_pt)
        ids_to_delete = [
            row["id"]
            for row in existing
            if str(row.get("normalized_macua") or "") not in allowed
        ]
        if ids_to_delete:
            client.table(VARIANTS_TABLE_NAME).delete().in_("id", ids_to_delete).execute()

    first_word = normalized[0]["pt"]
    principal, variants = _collect_variants_for_word(first_word)
    return {
        "saved": len(rows),
        "principal": principal,
        "variantes": variants,
    }


def delete_variant(pt: str, macua: str) -> int:
    client = get_client()
    norm_pt = _normalize_pt(pt)
    norm_macua = _normalize_macua(macua)
    resp = (
        client
        .table(VARIANTS_TABLE_NAME)
        .delete()
        .eq("normalized_pt", norm_pt)
        .eq("normalized_macua", norm_macua)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return len(data)


def delete_entry(pt: str) -> int:
    client = get_client()
    norm_pt = _normalize_pt(pt)
    resp = (
        client
        .table(VARIANTS_TABLE_NAME)
        .delete()
        .eq("normalized_pt", norm_pt)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    return len(data)

