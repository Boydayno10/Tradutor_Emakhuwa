import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from supabase_client_strict import load_resources


# Carrega recursos uma vez e mantém em cache em memória
_RES = load_resources()
_RAW_LEXICON: Dict[str, List[str]] = _RES.lexicon
_PRONOUNS = _RES.pronouns


def _normalize_pt(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _is_punctuation(tok: str) -> bool:
    return bool(re.fullmatch(r"[.,!?;:]+", tok))


"""Pipeline de tradução Emakua 

Suporta dois sentidos:
- PT -> Emakua (com correção ortográfica leve em PT)
- Emakua -> PT

Quando usado em modo "auto", detecta o sentido provável
da frase com base no léxico e pronomes disponíveis.
"""


# --- Índices ---

# PT -> Emakua
_LEXICON_PT: Dict[str, List[str]] = {}
_PRONOUN_PT: Dict[str, List[str]] = {}
_SPELL_VOCAB_PT: Dict[str, str] = {}

# Emakua -> PT
_LEXICON_EM: Dict[str, List[str]] = {}
_PRONOUN_EM: Dict[str, List[str]] = {}


# léxico
for pt_word, vals in _RAW_LEXICON.items():
    norm_pt = _normalize_pt(pt_word)
    if norm_pt not in _SPELL_VOCAB_PT:
        _SPELL_VOCAB_PT[norm_pt] = pt_word

    cleaned: List[str] = []
    for v in vals:
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s:
            continue
        cleaned.append(s)
    if not cleaned:
        continue

    # índice PT -> Emakua
    target_pt = _LEXICON_PT.setdefault(norm_pt, [])
    for c in cleaned:
        if c not in target_pt:
            target_pt.append(c)

    # índice Emakua -> PT (usamos lowercase simples)
    for em_form in cleaned:
        em_key = em_form.strip().lower()
        if not em_key:
            continue
        target_em = _LEXICON_EM.setdefault(em_key, [])
        if pt_word not in target_em:
            target_em.append(pt_word)


# pronomes
_pers = _PRONOUNS.get("personal", {})
_poss = _PRONOUNS.get("possessive", {})
for pt_pron, forms in {**_pers, **_poss}.items():
    norm_pt = _normalize_pt(pt_pron)
    em_forms = [f.strip() for f in forms if isinstance(f, str)]
    if not em_forms:
        continue

    # PT -> Emakua
    _PRONOUN_PT[norm_pt] = em_forms
    if norm_pt not in _SPELL_VOCAB_PT:
        _SPELL_VOCAB_PT[norm_pt] = pt_pron

    # Emakua -> PT
    for em_form in em_forms:
        em_key = em_form.strip().lower()
        if not em_key:
            continue
        target_em = _PRONOUN_EM.setdefault(em_key, [])
        if pt_pron not in target_em:
            target_em.append(pt_pron)


# --- Corretor ortográfico leve ---


def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def correct_spelling_pt(word: str) -> str:
    """Corretor ortográfico leve só para português."""

    norm = _normalize_pt(word)
    if norm in _SPELL_VOCAB_PT:
        return _SPELL_VOCAB_PT[norm]
    best_key: Optional[str] = None
    best_dist = math.inf
    for cand_norm in _SPELL_VOCAB_PT.keys():
        d = _levenshtein(norm, cand_norm)
        if d < best_dist:
            best_dist = d
            best_key = cand_norm
    if best_key is None:
        return word.lower()
    max_allowed = 2 if len(norm) <= 5 else 3
    if best_dist <= max_allowed:
        return _SPELL_VOCAB_PT[best_key]
    return word.lower()


def lookup_pt_to_em(word: str, missing_log: Optional[List[str]] = None) -> Dict:
    """Lookup de português para Emakua, com correção ortográfica PT."""

    corrected = correct_spelling_pt(word)
    norm = _normalize_pt(corrected)
    em_candidates: List[str] = []

    if norm in _PRONOUN_PT:
        em_candidates.extend(_PRONOUN_PT[norm])

    if norm in _LEXICON_PT:
        for v in _LEXICON_PT[norm]:
            if v not in em_candidates:
                em_candidates.append(v)

    found = bool(em_candidates)
    if not found and missing_log is not None:
        missing_log.append(word)

    return {
        "source": word,
        "normalized": norm,
        "candidates": em_candidates,
        "found": found,
    }


def lookup_em_to_pt(word: str, missing_log: Optional[List[str]] = None) -> Dict:
    """Lookup de Emakua para português (sem correção ortográfica)."""

    key = word.strip().lower()
    pt_candidates: List[str] = []

    if key in _PRONOUN_EM:
        pt_candidates.extend(_PRONOUN_EM[key])

    if key in _LEXICON_EM:
        for v in _LEXICON_EM[key]:
            if v not in pt_candidates:
                pt_candidates.append(v)

    found = bool(pt_candidates)
    if not found and missing_log is not None:
        missing_log.append(word)

    return {
        "source": word,
        "normalized": key,
        "candidates": pt_candidates,
        "found": found,
    }


# --- Tokenização e construção de frase ---


def _tokenize(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    return [t for t in text.split() if t]


def _build_sentence_from_lookup(tokens: List[str], direction: str) -> str:
    missing: List[str] = []
    out_tokens: List[str] = []

    for tok in tokens:
        if _is_punctuation(tok):
            out_tokens.append(tok)
            continue

        if direction == "pt_to_em":
            info = lookup_pt_to_em(tok, missing)
        else:  # em_to_pt
            info = lookup_em_to_pt(tok, missing)

        candidates = info["candidates"]
        if candidates:
            out_tokens.append(candidates[0])
        else:
            out_tokens.append(tok)

    sentence = " ".join(out_tokens)
    sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def _count_known_tokens(tokens: List[str]) -> Tuple[int, int]:
    """Conta quantos tokens parecem PT e quantos parecem Emakua.

    Usa apenas presença nos índices (sem correção),
    para não distorcer a detecção de língua.
    """

    pt_count = 0
    em_count = 0
    for tok in tokens:
        if _is_punctuation(tok):
            continue
        norm_pt = _normalize_pt(tok)
        em_key = tok.strip().lower()

        if norm_pt in _LEXICON_PT or norm_pt in _PRONOUN_PT:
            pt_count += 1
        if em_key in _LEXICON_EM or em_key in _PRONOUN_EM:
            em_count += 1
    return pt_count, em_count


def _detect_direction(tokens: List[str]) -> str:
    """Detecta automaticamente se a frase é PT ou Emakua."""

    pt_count, em_count = _count_known_tokens(tokens)
    if em_count > pt_count:
        return "em_to_pt"
    # empate ou mais PT -> assume PT -> Emakua (uso mais comum)
    return "pt_to_em"


def translate_pt_to_em(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    tokens = _tokenize(text)
    return _build_sentence_from_lookup(tokens, "pt_to_em")


def translate_em_to_pt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    tokens = _tokenize(text)
    return _build_sentence_from_lookup(tokens, "em_to_pt")


def translate(text: str, direction: str = "auto") -> str:
    """Tradução de texto entre PT e Emakua.

    direction:
      - "pt_to_em": força PT -> Emakua
      - "em_to_pt": força Emakua -> PT
      - "auto"   : detecta automaticamente
    """

    text = text.strip()
    if not text:
        return ""

    tokens = _tokenize(text)

    if direction == "pt_to_em":
        return _build_sentence_from_lookup(tokens, "pt_to_em")
    if direction == "em_to_pt":
        return _build_sentence_from_lookup(tokens, "em_to_pt")

    auto_dir = _detect_direction(tokens)
    return _build_sentence_from_lookup(tokens, auto_dir)
