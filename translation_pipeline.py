import math
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from supabase_client_strict import EmakuaResources, load_resources


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

Os dados (léxico, gramática, pronomes) são carregados
dinamicamente do Supabase em cada requisição, usando um
cache em memória com TTL fornecido por supabase_client_strict.
Nenhum JSON é carregado de forma global na importação do módulo.
"""


def _build_indexes(resources: EmakuaResources) -> Tuple[
    Dict[str, List[str]],  # lexicon_pt
    Dict[str, List[str]],  # pronoun_pt
    Dict[str, str],        # spell_vocab_pt
    Dict[str, List[str]],  # lexicon_em
    Dict[str, List[str]],  # pronoun_em
]:
    """Constroi todos os índices necessários a partir dos recursos atuais.

    Esta função é chamada a partir das funções públicas de tradução,
    garantindo que os dados sejam sempre derivados dos recursos
    obtidos dinamicamente (com TTL) do Supabase.
    """

    raw_lexicon: Dict[str, List[str]] = resources.lexicon
    pronouns = resources.pronouns

    lexicon_pt: Dict[str, List[str]] = {}
    pronoun_pt: Dict[str, List[str]] = {}
    spell_vocab_pt: Dict[str, str] = {}
    lexicon_em: Dict[str, List[str]] = {}
    pronoun_em: Dict[str, List[str]] = {}

    # léxico
    for pt_word, vals in raw_lexicon.items():
        if isinstance(vals, str):
            vals = [vals]
        elif not isinstance(vals, list):
            continue
        norm_pt = _normalize_pt(pt_word)
        if norm_pt not in spell_vocab_pt:
            spell_vocab_pt[norm_pt] = pt_word

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
        target_pt = lexicon_pt.setdefault(norm_pt, [])
        for c in cleaned:
            if c not in target_pt:
                target_pt.append(c)

        # índice Emakua -> PT (usamos lowercase simples)
        for em_form in cleaned:
            em_key = em_form.strip().lower()
            if not em_key:
                continue
            target_em = lexicon_em.setdefault(em_key, [])
            if pt_word not in target_em:
                target_em.append(pt_word)

    # pronomes
    _pers = pronouns.get("personal", {})
    _poss = pronouns.get("possessive", {})
    for pt_pron, forms in {**_pers, **_poss}.items():
        norm_pt = _normalize_pt(pt_pron)
        em_forms = [f.strip() for f in forms if isinstance(f, str)]
        if not em_forms:
            continue

        # PT -> Emakua
        pronoun_pt[norm_pt] = em_forms
        if norm_pt not in spell_vocab_pt:
            spell_vocab_pt[norm_pt] = pt_pron

        # Emakua -> PT
        for em_form in em_forms:
            em_key = em_form.strip().lower()
            if not em_key:
                continue
            target_em = pronoun_em.setdefault(em_key, [])
            if pt_pron not in target_em:
                target_em.append(pt_pron)

    return lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em


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


def correct_spelling_pt(word: str, spell_vocab_pt: Dict[str, str]) -> str:
    """Corretor ortográfico leve só para português."""

    norm = _normalize_pt(word)
    if norm in spell_vocab_pt:
        return spell_vocab_pt[norm]
    best_key: Optional[str] = None
    best_dist = math.inf
    for cand_norm in spell_vocab_pt.keys():
        d = _levenshtein(norm, cand_norm)
        if d < best_dist:
            best_dist = d
            best_key = cand_norm
    if best_key is None:
        return word.lower()
    max_allowed = 2 if len(norm) <= 5 else 3
    if best_dist <= max_allowed:
        return spell_vocab_pt[best_key]
    return word.lower()


def lookup_pt_to_em(
    word: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    missing_log: Optional[List[str]] = None,
) -> Dict:
    """Lookup de português para Emakua, com correção ortográfica PT."""

    corrected = correct_spelling_pt(word, spell_vocab_pt)
    norm = _normalize_pt(corrected)
    em_candidates: List[str] = []

    if norm in pronoun_pt:
        em_candidates.extend(pronoun_pt[norm])

    if norm in lexicon_pt:
        for v in lexicon_pt[norm]:
            if v not in em_candidates:
                em_candidates.append(v)
    # Garante no máximo 4 traduções por palavra
    if len(em_candidates) > 4:
        em_candidates = em_candidates[:4]
    found = bool(em_candidates)
    if not found and missing_log is not None:
        missing_log.append(word)

    return {
        "source": word,
        "normalized": norm,
        "candidates": em_candidates,
        "found": found,
    }


def lookup_em_to_pt(
    word: str,
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    missing_log: Optional[List[str]] = None,
) -> Dict:
    """Lookup de Emakua para português (sem correção ortográfica)."""

    key = word.strip().lower()
    pt_candidates: List[str] = []

    if key in pronoun_em:
        pt_candidates.extend(pronoun_em[key])

    if key in lexicon_em:
        for v in lexicon_em[key]:
            if v not in pt_candidates:
                pt_candidates.append(v)
    # Garante no máximo 4 traduções por palavra
    if len(pt_candidates) > 4:
        pt_candidates = pt_candidates[:4]
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


def _build_sentence_from_lookup(
    tokens: List[str],
    direction: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
) -> str:
    missing: List[str] = []
    out_tokens: List[str] = []
    # Caso especial: entrada de única palavra (sem pontuação)
    # Retorna até 4 traduções possíveis da palavra.
    if len(tokens) == 1 and not _is_punctuation(tokens[0]):
        tok = tokens[0]
        if direction == "pt_to_em":
            info = lookup_pt_to_em(tok, lexicon_pt, pronoun_pt, spell_vocab_pt, missing)
        else:  # em_to_pt
            info = lookup_em_to_pt(tok, lexicon_em, pronoun_em, missing)

        candidates = info["candidates"][:4]
        if candidates:
            sentence = ", ".join(candidates)
            if sentence:
                sentence = sentence[0].upper() + sentence[1:]
            return sentence

        # Sem candidatos conhecidos, devolve a própria palavra
        return tok
    for tok in tokens:
        if _is_punctuation(tok):
            out_tokens.append(tok)
            continue

        if direction == "pt_to_em":
            info = lookup_pt_to_em(tok, lexicon_pt, pronoun_pt, spell_vocab_pt, missing)
        else:  # em_to_pt
            info = lookup_em_to_pt(tok, lexicon_em, pronoun_em, missing)

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


def _count_known_tokens(
    tokens: List[str],
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
) -> Tuple[int, int]:
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

        if norm_pt in lexicon_pt or norm_pt in pronoun_pt:
            pt_count += 1
        if em_key in lexicon_em or em_key in pronoun_em:
            em_count += 1
    return pt_count, em_count


def _detect_direction(
    tokens: List[str],
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
) -> str:
    """Detecta automaticamente se a frase é PT ou Emakua."""

    pt_count, em_count = _count_known_tokens(tokens, lexicon_pt, pronoun_pt, lexicon_em, pronoun_em)
    if em_count > pt_count:
        return "em_to_pt"
    # empate ou mais PT -> assume PT -> Emakua (uso mais comum)
    return "pt_to_em"


def translate_pt_to_em(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em = _build_indexes(resources)

    tokens = _tokenize(text)
    return _build_sentence_from_lookup(
        tokens,
        "pt_to_em",
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
    )


def translate_em_to_pt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em = _build_indexes(resources)

    tokens = _tokenize(text)
    return _build_sentence_from_lookup(
        tokens,
        "em_to_pt",
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
    )


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

    # Carrega recursos dinamicamente (com cache TTL) a cada chamada,
    # garantindo que a consulta ao Supabase faça parte do fluxo da rota.
    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em = _build_indexes(resources)

    tokens = _tokenize(text)

    if direction == "pt_to_em":
        return _build_sentence_from_lookup(
            tokens,
            "pt_to_em",
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
        )
    if direction == "em_to_pt":
        return _build_sentence_from_lookup(
            tokens,
            "em_to_pt",
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
        )

    auto_dir = _detect_direction(tokens, lexicon_pt, pronoun_pt, lexicon_em, pronoun_em)
    return _build_sentence_from_lookup(
        tokens,
        auto_dir,
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
    )
