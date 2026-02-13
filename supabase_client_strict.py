import os
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client, create_client


SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_ANON_KEY")

TABLE_NAME = "emakua_ml_resources"
VARIANTS_TABLE_NAME = os.environ.get(
    "EMAKUA_VARIANTS_TABLE_NAME",
    "emakua_translation_variants",
)
PHRASE_MEMORY_TABLE_NAME = os.environ.get(
    "EMAKUA_PHRASE_MEMORY_TABLE_NAME",
    "emakua_phrase_memory",
)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY/ANON_KEY precisam estar definidos no ambiente.")

_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


@dataclass
class EmakuaResources:
    grammar: Dict[str, Any]
    pronouns: Dict[str, Any]
    lexicon: Dict[str, Any]


# TTL (em segundos) configurável via variável de ambiente.
#
# Por padrão usamos 0 para garantir que **toda** requisição consulte
# diretamente o Supabase, sem reutilizar dados em cache.
# Se quiser habilitar cache, defina EMAKUA_CACHE_TTL_SECONDS>0.
_CACHE_TTL_SECONDS: int = int(os.environ.get("EMAKUA_CACHE_TTL_SECONDS", "0"))

# Cache em memória separado por recurso: nome -> (timestamp, dados)
_resource_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _normalize_text(text: str) -> str:
    value = (text or "").strip().lower()
    value = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def _fetch_resource(name: str) -> Dict[str, Any]:
    """Busca o JSON bruto no Supabase, sem cache."""
    resp = (
        _client
        .table(TABLE_NAME)
        .select("metadata")
        .eq("name", name)
        .execute()
    )
    data = getattr(resp, "data", None)
    if not data:
        raise RuntimeError(f"Recurso {name} não encontrado na tabela {TABLE_NAME}.")
    return data[0]["metadata"]


def _fetch_all_rows(
    table: str,
    select_expr: str,
    batch_size: int = 1000,
    order_by: Optional[str] = None,
    order_desc: bool = False,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start = 0
    while True:
        end = start + batch_size - 1
        query = (
            _client
            .table(table)
            .select(select_expr)
        )
        if order_by:
            query = query.order(order_by, desc=order_desc)
        resp = query.range(start, end).execute()
        data = getattr(resp, "data", None) or []
        if not data:
            break
        rows.extend(data)
        if len(data) < batch_size:
            break
        start += batch_size
    return rows


def _get_with_ttl(name: str) -> Dict[str, Any]:
    """Retorna o recurso do cache em memória com TTL ou recarrega do Supabase."""

    # Se TTL <= 0, desabilita completamente o cache: sempre consulta Supabase.
    if _CACHE_TTL_SECONDS <= 0:
        return _fetch_resource(name)

    now = time.time()
    cached = _resource_cache.get(name)
    if cached is not None:
        ts, data = cached
        if now - ts < _CACHE_TTL_SECONDS:
            return data

    data = _fetch_resource(name)
    _resource_cache[name] = (now, data)
    return data


def get_pt_emakua_lexicon() -> Dict[str, Any]:
    """Retorna o léxico pt_emakua_lexicon com cache em memória e TTL."""

    raw = _get_with_ttl("pt_emakua_lexicon.json")
    if not isinstance(raw, dict):
        raise RuntimeError("Léxico inválido (metadata não é um objeto).")

    # Normalize values to List[str].
    # This keeps the translation pipeline stable even if some entries were
    # saved as a single string (e.g. via correction UI).
    normalized: Dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not k.strip():
            continue
        if isinstance(v, str):
            s = v.strip()
            if not s:
                continue
            normalized[k] = [s]
            continue
        if isinstance(v, list):
            cleaned = [
                item.strip()
                for item in v
                if isinstance(item, str) and item.strip()
            ]
            if not cleaned:
                continue
            normalized[k] = cleaned
            continue
        # Ignore other types (numbers/objects) to avoid corrupting indexes.

    # Merge correction variants stored as independent rows.
    # Table schema expected: (pt text, macua text, ...).
    # If table is absent, we keep the legacy metadata-only behavior.
    try:
        variant_rows = _fetch_all_rows(
            VARIANTS_TABLE_NAME,
            "pt,macua,normalized_pt,created_at,updated_at",
            order_by="updated_at",
            order_desc=True,
        )
    except Exception:
        variant_rows = []

    # Ordenacao deterministica para evitar retorno aleatorio de variantes.
    variant_rows = sorted(
        variant_rows,
        key=lambda row: (
            str((row or {}).get("normalized_pt") or _normalize_text(str((row or {}).get("pt") or ""))),
            str((row or {}).get("updated_at") or ""),
            str((row or {}).get("created_at") or ""),
            str((row or {}).get("macua") or "").lower(),
        ),
        reverse=True,
    )

    # Preserve existing key casing where possible.
    key_by_norm = {_normalize_text(k): k for k in normalized.keys()}
    for row in variant_rows:
        pt = str((row or {}).get("pt") or "").strip()
        macua = str((row or {}).get("macua") or "").strip()
        if not pt or not macua:
            continue

        norm_pt = _normalize_text(pt)
        target_key = key_by_norm.get(norm_pt, pt)
        if norm_pt not in key_by_norm:
            key_by_norm[norm_pt] = target_key

        bucket = normalized.setdefault(target_key, [])
        if not any(existing.lower() == macua.lower() for existing in bucket):
            bucket.append(macua)

    return normalized


def get_emakua_grammar() -> Dict[str, Any]:
    """Retorna a gramática emakua_grammar com cache em memória e TTL."""

    return _get_with_ttl("emakua_grammar.json")


def get_emakua_pronouns() -> Dict[str, Any]:
    """Retorna os pronomes emakua_pronouns com cache em memória e TTL."""

    return _get_with_ttl("emakua_pronouns.json")


def load_resources() -> EmakuaResources:
    """Carrega todos os recursos necessários para o pipeline de tradução.

    A chamada continua sendo simples para o restante do código, mas por baixo
    cada tipo de dado é carregado com cache em memória e TTL.
    """

    grammar = get_emakua_grammar()
    pronouns = get_emakua_pronouns()
    lexicon = get_pt_emakua_lexicon()

    return EmakuaResources(grammar=grammar, pronouns=pronouns, lexicon=lexicon)


def get_client() -> Client:
    """Exposes the shared Supabase client for server-side service modules."""
    return _client


def get_phrase_memory_preferences(normalized_source_phrase: str) -> List[Dict[str, Any]]:
    """Returns per-token preferences learned for a specific normalized PT phrase.

    Output rows contain at least:
      - token_pt
      - normalized_token_pt
      - selected_macua
      - normalized_selected_macua
      - position_index
    """
    phrase = (normalized_source_phrase or "").strip()
    if not phrase:
        return []
    try:
        resp = (
            _client
            .table(PHRASE_MEMORY_TABLE_NAME)
            .select(
                "token_pt,normalized_token_pt,selected_macua,normalized_selected_macua,position_index"
            )
            .eq("normalized_source_phrase", phrase)
            .order("position_index")
            .execute()
        )
        return getattr(resp, "data", None) or []
    except Exception:
        return []
