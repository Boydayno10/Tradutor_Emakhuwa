from __future__ import annotations

from typing import Any, Dict, List, Tuple

from supabase_client_strict import VARIANTS_TABLE_NAME, get_client
from translation_pipeline import (
    _build_indexes,
    _is_punctuation,
    _levenshtein,
    _normalize_pt,
    _tokenize,
    load_resources,
    lookup_pt_to_em,
)

LEXICON_RESOURCE_TABLE = "emakua_ml_resources"
LEXICON_RESOURCE_NAME = "pt_emakua_lexicon.json"
PHRASE_MEMORY_TABLE = "emakua_phrase_memory"


def _normalize_macua(text: str) -> str:
    return (text or "").strip().lower()


def _canonicalize_phrase_pt(text: str) -> str:
    tokens = _tokenize(text or "")
    out: List[str] = []
    for tok in tokens:
        if _is_punctuation(tok):
            out.append(tok)
        else:
            out.append(_normalize_pt(tok))
    raw = " ".join(out)
    return raw.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?").replace(" ;", ";").replace(" :", ":")


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


def _parse_metadata_value(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if "," in raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [raw]
    return []


def _load_lexicon_resource() -> Tuple[str, Dict[str, Any]]:
    client = get_client()
    resp = (
        client
        .table(LEXICON_RESOURCE_TABLE)
        .select("id,metadata")
        .eq("name", LEXICON_RESOURCE_NAME)
        .limit(1)
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    if not rows:
        raise RuntimeError(f"Recurso {LEXICON_RESOURCE_NAME} nao encontrado em {LEXICON_RESOURCE_TABLE}")
    row = rows[0]
    resource_id = str(row.get("id") or "").strip()
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return resource_id, dict(metadata)


def _save_lexicon_metadata(resource_id: str, metadata: Dict[str, Any]) -> None:
    client = get_client()
    (
        client
        .table(LEXICON_RESOURCE_TABLE)
        .update({"metadata": metadata})
        .eq("id", resource_id)
        .execute()
    )


def _metadata_keys_by_norm_pt(metadata: Dict[str, Any], norm_pt: str) -> List[str]:
    keys: List[str] = []
    for key in metadata.keys():
        if _normalize_pt(str(key)) == norm_pt:
            keys.append(str(key))
    return keys


def _sync_metadata_entry(pt: str, macuas: List[str]) -> None:
    resource_id, metadata = _load_lexicon_resource()
    norm_pt = _normalize_pt(pt)
    matching_keys = _metadata_keys_by_norm_pt(metadata, norm_pt)

    for key in matching_keys:
        if key != pt:
            metadata.pop(key, None)

    deduped_macuas: List[str] = []
    seen = set()
    for item in macuas:
        value = str(item).strip()
        if not value:
            continue
        k = value.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped_macuas.append(value)

    if deduped_macuas:
        metadata[pt] = deduped_macuas
    else:
        metadata.pop(pt, None)

    _save_lexicon_metadata(resource_id, metadata)


def _remove_metadata_variant(pt: str, macua: str) -> None:
    resource_id, metadata = _load_lexicon_resource()
    norm_pt = _normalize_pt(pt)
    norm_macua = _normalize_macua(macua)

    changed = False
    for key in list(metadata.keys()):
        if _normalize_pt(str(key)) != norm_pt:
            continue
        current = _parse_metadata_value(metadata.get(key))
        filtered = [item for item in current if _normalize_macua(item) != norm_macua]
        if filtered:
            metadata[key] = filtered
        else:
            metadata.pop(key, None)
        changed = True

    if changed:
        _save_lexicon_metadata(resource_id, metadata)


def _remove_metadata_entry(pt: str) -> None:
    resource_id, metadata = _load_lexicon_resource()
    norm_pt = _normalize_pt(pt)
    changed = False
    for key in list(metadata.keys()):
        if _normalize_pt(str(key)) == norm_pt:
            metadata.pop(key, None)
            changed = True
    if changed:
        _save_lexicon_metadata(resource_id, metadata)


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


def _fetch_phrase_memory_rows(normalized_phrase: str) -> List[Dict[str, Any]]:
    client = get_client()
    try:
        resp = (
            client
            .table(PHRASE_MEMORY_TABLE)
            .select("position_index,selected_macua")
            .eq("normalized_source_phrase", normalized_phrase)
            .order("position_index")
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception:
        return []


def _fetch_rows_by_fuzzy_norm_pt(norm_pt: str, max_distance: int = 2) -> List[Dict[str, Any]]:
    client = get_client()
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


def _collect_candidates_for_token(
    token_pt: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    fuzzy: bool = False,
) -> List[str]:
    info = lookup_pt_to_em(token_pt, lexicon_pt, pronoun_pt, spell_vocab_pt)
    norm_pt = str(info.get("normalized") or _normalize_pt(token_pt))

    candidates: List[str] = []
    for c in info.get("candidates", []) or []:
        s = str(c).strip()
        if s and s.lower() not in {v.lower() for v in candidates}:
            candidates.append(s)

    dist_limit = _pt_norm_distance_limit(norm_pt)
    for pt_norm, macuas in lexicon_pt.items():
        is_match = pt_norm == norm_pt
        if fuzzy and not is_match:
            if abs(len(pt_norm) - len(norm_pt)) <= dist_limit and _levenshtein(pt_norm, norm_pt) <= dist_limit:
                is_match = True
        if not is_match:
            continue
        for macua in macuas:
            s = str(macua).strip()
            if s and s.lower() not in {v.lower() for v in candidates}:
                candidates.append(s)

    rows = _fetch_rows_by_norm_pt(norm_pt)
    if fuzzy and not rows:
        rows = _fetch_rows_by_fuzzy_norm_pt(norm_pt, max_distance=dist_limit)
    for row in rows:
        s = str((row or {}).get("macua") or "").strip()
        if s and s.lower() not in {v.lower() for v in candidates}:
            candidates.append(s)

    return candidates


def _collect_variants_for_word(word: str) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, _, _ = _build_indexes(resources)

    candidates = _collect_candidates_for_token(word, lexicon_pt, pronoun_pt, spell_vocab_pt, fuzzy=True)
    pairs = [{"pt": word.strip(), "macua": c} for c in candidates]

    deduped = _dedupe_pairs(pairs)
    principal = deduped[0] if deduped else {"pt": word.strip(), "macua": ""}
    return principal, deduped


def _join_phrase_tokens(parts: List[str]) -> str:
    raw = " ".join(parts)
    raw = raw.replace(" ,", ",").replace(" .", ".").replace(" !", "!")
    raw = raw.replace(" ?", "?").replace(" ;", ";").replace(" :", ":")
    return raw.strip()


def get_phrase_correction_payload(texto: str) -> Dict[str, Any]:
    phrase = (texto or "").strip()
    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, _, _ = _build_indexes(resources)

    tokens = _tokenize(phrase)
    memory_rows = _fetch_phrase_memory_rows(_canonicalize_phrase_pt(phrase))
    memory_map = {}
    for row in memory_rows:
        try:
            pos = int(row.get("position_index") or 0)
        except Exception:
            pos = 0
        sel = str(row.get("selected_macua") or "").strip()
        if pos > 0 and sel:
            memory_map[pos] = sel

    words_payload: List[Dict[str, Any]] = []
    composed_parts: List[str] = []

    pos_counter = 0
    for token in tokens:
        if _is_punctuation(token):
            composed_parts.append(token)
            words_payload.append(
                {
                    "posicao": pos_counter,
                    "pt": token,
                    "pontuacao": True,
                    "variantes": [{"macua": token, "selecionada": True}],
                    "selecionada": token,
                }
            )
            continue

        pos_counter += 1
        candidates = _collect_candidates_for_token(token, lexicon_pt, pronoun_pt, spell_vocab_pt, fuzzy=False)
        selected = candidates[0] if candidates else token
        preferred = memory_map.get(pos_counter)
        if preferred:
            preferred_match = next(
                (c for c in candidates if c.lower() == preferred.lower()),
                None,
            )
            if preferred_match:
                selected = preferred_match
        composed_parts.append(selected)

        words_payload.append(
            {
                "posicao": pos_counter,
                "pt": token,
                "pontuacao": False,
                "variantes": [
                    {"macua": c, "selecionada": c == selected}
                    for c in candidates
                ]
                if candidates
                else [{"macua": token, "selecionada": True}],
                "selecionada": selected,
            }
        )

    frase_montada = _join_phrase_tokens(composed_parts)
    if frase_montada:
        frase_montada = frase_montada[0].upper() + frase_montada[1:]

    return {
        "entrada": phrase,
        "frase_montada": frase_montada,
        "palavras": words_payload,
    }


def save_phrase_learning(payload: Dict[str, Any]) -> Dict[str, Any]:
    client = get_client()
    frase_original = str(payload.get("frase_original") or "").strip()
    palavras = payload.get("palavras") or []
    if not isinstance(palavras, list) or not palavras:
        raise RuntimeError("Campo 'palavras' invalido")

    canonical_phrase = _canonicalize_phrase_pt(frase_original)

    phrase_rows: List[Dict[str, Any]] = []
    reconstructed_parts: List[str] = []

    variants_to_upsert: List[Dict[str, str]] = []

    ordered = sorted(
        [p for p in palavras if isinstance(p, dict)],
        key=lambda x: int(x.get("posicao") or 0),
    )

    visible_pos = 0
    for item in ordered:
        pt = str(item.get("pt") or "").strip()
        if not pt:
            continue

        is_punctuation = bool(item.get("pontuacao")) or _is_punctuation(pt)
        selected = str(item.get("selecionada") or "").strip()
        variantes = item.get("variantes") or []
        if not isinstance(variantes, list):
            variantes = []

        if is_punctuation:
            reconstructed_parts.append(pt)
            continue

        visible_pos += 1

        all_variants: List[str] = []
        for v in variantes:
            if isinstance(v, dict):
                val = str(v.get("macua") or "").strip()
            else:
                val = str(v).strip()
            if val and val.lower() not in {x.lower() for x in all_variants}:
                all_variants.append(val)

        if selected and selected.lower() not in {x.lower() for x in all_variants}:
            all_variants.insert(0, selected)

        if not selected:
            selected = all_variants[0] if all_variants else pt
        reconstructed_parts.append(selected)

        if all_variants:
            _sync_metadata_entry(pt, all_variants)
            for macua in all_variants:
                variants_to_upsert.append({"pt": pt, "macua": macua})

        phrase_rows.append(
            {
                "source_phrase": frase_original,
                "normalized_source_phrase": canonical_phrase,
                "token_pt": pt,
                "normalized_token_pt": _normalize_pt(pt),
                "selected_macua": selected,
                "normalized_selected_macua": _normalize_macua(selected),
                "position_index": visible_pos,
                "phrase_length": len([p for p in ordered if not bool(p.get("pontuacao")) and not _is_punctuation(str(p.get("pt") or ""))]),
                "metadata": {
                    "variantes": all_variants,
                },
            }
        )

    if variants_to_upsert:
        upsert_variants(variants_to_upsert)

    phrase_memory_saved = False
    phrase_memory_error = ""

    # Replace memory for this phrase key. If this table is unavailable,
    # we still keep per-word variants saved to avoid losing learning.
    try:
        (
            client
            .table(PHRASE_MEMORY_TABLE)
            .delete()
            .eq("normalized_source_phrase", canonical_phrase)
            .execute()
        )

        if phrase_rows:
            (
                client
                .table(PHRASE_MEMORY_TABLE)
                .insert(phrase_rows)
                .execute()
            )
        phrase_memory_saved = True
    except Exception as exc:
        phrase_memory_saved = False
        phrase_memory_error = str(exc)

    frase_montada = _join_phrase_tokens(reconstructed_parts)
    if frase_montada:
        frase_montada = frase_montada[0].upper() + frase_montada[1:]

    return {
        "saved": len(phrase_rows),
        "frase_montada": frase_montada,
        "normalizada": canonical_phrase,
        "phrase_memory_saved": phrase_memory_saved,
        "phrase_memory_error": phrase_memory_error,
    }


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

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        norm_pt = row["normalized_pt"]
        bucket = grouped.setdefault(norm_pt, {"pt": row["pt"], "macuas": []})
        bucket["macuas"].append(row["macua"])
    for payload in grouped.values():
        _sync_metadata_entry(payload["pt"], payload["macuas"])

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
    _remove_metadata_variant(pt, macua)
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
    _remove_metadata_entry(pt)
    return len(data)
