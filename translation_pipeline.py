import math
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from supabase_client_strict import (
    EmakuaResources,
    get_phrase_memory_preferences,
    load_resources,
)


def _normalize_pt(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def _is_punctuation(tok: str) -> bool:
    return bool(re.fullmatch(r"[.,!?;:]+", tok))


_OMIT_PT_TOKENS = {"o", "a", "os", "as", "da"}
_CONNECTOR_PT_TOKENS = {"e"}
_DEMONSTRATIVE_SUFFIX_BY_TOKEN = {
    "esse": "owo",
    "essa": "ola",
}
_POSSESSIVE_SUFFIX_BY_PRONOUN = {
    # Regra 2 (adotado: sufixo "ka" para meu/minha)
    "meu": "ka",
    "minha": "ka",
    "meus": "ka",
    "minhas": "ka",
    # Regra 3
    "nosso": "hum",
    "nossa": "hum",
    "nossos": "hum",
    "nossas": "hum",
    # Regra 4
    "seu": "nho",
    "sua": "nho",
    "seus": "nho",
    "suas": "nho",
    "teu": "nho",
    "tua": "nho",
    "teus": "nho",
    "tuas": "nho",
    # Regra 5
    "vosso": "nhussa",
    "vossa": "nhussa",
    "vossos": "nhussa",
    "vossas": "nhussa",
}
_PT_ADJECTIVE_HINTS = {
    "bonito",
    "bonita",
    "bonitos",
    "bonitas",
    "caro",
    "cara",
    "caros",
    "caras",
    "grande",
    "grandes",
    "pequeno",
    "pequena",
    "pequenos",
    "pequenas",
    "novo",
    "nova",
    "novos",
    "novas",
    "velho",
    "velha",
    "velhos",
    "velhas",
    "bom",
    "boa",
    "bons",
    "boas",
    "mau",
    "ma",
    "maus",
    "mas",
    "lindo",
    "linda",
    "lindos",
    "lindas",
    "feio",
    "feia",
    "feios",
    "feias",
}

_PT_QUESTION_HINTS = {
    "quem",
    "que",
    "qual",
    "quais",
    "quando",
    "onde",
    "aonde",
    "como",
    "quanto",
    "quantos",
    "quanta",
    "quantas",
    "porque",
    "por",
}

_PT_EXCLAMATION_HINTS = {
    "uau",
    "nossa",
    "socorro",
    "cuidado",
    "parabens",
    "incrivel",
    "fantastico",
    "otimo",
    "excelente",
}
_POSSESSIVE_LIST_TRIGGER_PRONOUNS = {"meu", "minha", "meus", "minhas"}
_COPULA_PT_TOKENS = {"e"}
_PT_LIVING_HINTS = {
    "homem",
    "mulher",
    "pessoa",
    "pessoas",
    "crianca",
    "menino",
    "menina",
    "adulto",
    "idoso",
    "animal",
    "animais",
    "cao",
    "cachorro",
    "cadela",
    "gato",
    "gata",
    "passaro",
    "vaca",
    "boi",
    "cabra",
    "galinha",
    "planta",
    "plantas",
    "arvore",
    "arvores",
    "flor",
    "flores",
}
_PT_PERSON_ANIMAL_HINTS = {
    "homem",
    "mulher",
    "pessoa",
    "pessoas",
    "crianca",
    "menino",
    "menina",
    "adulto",
    "idoso",
    "animal",
    "animais",
    "cao",
    "cachorro",
    "cadela",
    "gato",
    "gata",
    "passaro",
    "vaca",
    "boi",
    "cabra",
    "galinha",
    "peixe",
}


def _dedupe_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        s = str(item or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _extract_grammar_profile(grammar: Dict[str, Any]) -> Dict[str, Any]:
    phonetics = grammar.get("phonetics", {}) if isinstance(grammar, dict) else {}
    digraphs = phonetics.get("digraphs", {}) if isinstance(phonetics, dict) else {}
    digraph_set = {
        str(k).strip().lower()
        for k in digraphs.keys()
        if isinstance(k, str) and str(k).strip()
    }
    vowels = {"a", "e", "i", "o", "u"}
    return {
        "digraphs": digraph_set,
        "vowels": vowels,
    }


def _extract_pronoun_pt_map(pronouns: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    if not isinstance(pronouns, dict):
        return result

    blocks: List[Dict[str, Any]] = []
    for key in ("personal", "possessive"):
        value = pronouns.get(key)
        if isinstance(value, dict):
            blocks.append(value)

    for block in blocks:
        for pt_pron, forms in block.items():
            norm_pt = _normalize_pt(str(pt_pron))
            if not norm_pt:
                continue
            if isinstance(forms, str):
                raw_forms = [forms]
            elif isinstance(forms, list):
                raw_forms = [f for f in forms if isinstance(f, str)]
            else:
                continue
            cleaned = _dedupe_keep_order([str(f).strip() for f in raw_forms])
            if not cleaned:
                continue
            current = result.get(norm_pt, {})
            existing_forms = current.get("forms", []) if isinstance(current, dict) else []
            result[norm_pt] = {
                "pt": str(pt_pron).strip() or norm_pt,
                "forms": _dedupe_keep_order(existing_forms + cleaned),
            }

    return result


def _candidate_quality_score_emakua(form: str, grammar_profile: Dict[str, Any]) -> int:
    s = (form or "").strip().lower()
    if not s:
        return -10_000

    score = 0
    vowels: Set[str] = grammar_profile.get("vowels", {"a", "e", "i", "o", "u"})
    digraphs: Set[str] = grammar_profile.get("digraphs", set())

    # Prefer forms aligned with orthography notes (k in place of c/q).
    if "c" not in s and "q" not in s:
        score += 3

    # Grammar note: words usually end in vowel.
    if s[-1:] in vowels:
        score += 2

    # Slight bonus for recognized Emakua digraph usage.
    for dg in digraphs:
        if dg and dg in s:
            score += 1

    # Penalize noisy forms with many symbols.
    if re.search(r"[^a-z' -]", s):
        score -= 3

    return score


def _rank_candidates_for_emakua(
    candidates: List[str],
    grammar_profile: Dict[str, Any],
) -> List[str]:
    unique = _dedupe_keep_order(candidates)
    ranked = sorted(
        enumerate(unique),
        key=lambda pair: (
            -_candidate_quality_score_emakua(pair[1], grammar_profile),
            pair[0],
            pair[1].lower(),
        ),
    )
    return [word for _, word in ranked]


"""Pipeline de traduÃ§Ã£o Emakua

Suporta dois sentidos:
- PT -> Emakua (com correÃ§Ã£o ortogrÃ¡fica leve em PT)
- Emakua -> PT

Quando usado em modo "auto", detecta o sentido provÃ¡vel
da frase com base no lÃ©xico e pronomes disponÃ­veis.

Os dados (lÃ©xico, gramÃ¡tica, pronomes) sÃ£o carregados
dinamicamente do Supabase em cada requisiÃ§Ã£o, usando um
cache em memÃ³ria com TTL fornecido por supabase_client_strict.
Nenhum JSON Ã© carregado de forma global na importaÃ§Ã£o do mÃ³dulo.
"""


def _build_indexes(resources: EmakuaResources) -> Tuple[
    Dict[str, List[str]],  # lexicon_pt
    Dict[str, List[str]],  # pronoun_pt
    Dict[str, str],        # spell_vocab_pt
    Dict[str, List[str]],  # lexicon_em
    Dict[str, List[str]],  # pronoun_em
    Dict[str, Any],        # grammar_profile
]:
    """Constroi todos os Ã­ndices necessÃ¡rios a partir dos recursos atuais.

    Esta funÃ§Ã£o Ã© chamada a partir das funÃ§Ãµes pÃºblicas de traduÃ§Ã£o,
    garantindo que os dados sejam sempre derivados dos recursos
    obtidos dinamicamente (com TTL) do Supabase.
    """

    raw_lexicon: Dict[str, List[str]] = resources.lexicon
    pronouns = resources.pronouns
    grammar_profile = _extract_grammar_profile(resources.grammar)

    lexicon_pt: Dict[str, List[str]] = {}
    pronoun_pt: Dict[str, List[str]] = {}
    spell_vocab_pt: Dict[str, str] = {}
    lexicon_em: Dict[str, List[str]] = {}
    pronoun_em: Dict[str, List[str]] = {}

    # lÃ©xico
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

        # Ã­ndice PT -> Emakua
        target_pt = lexicon_pt.setdefault(norm_pt, [])
        for c in cleaned:
            if c not in target_pt:
                target_pt.append(c)

        # Ã­ndice Emakua -> PT (usamos lowercase simples)
        for em_form in cleaned:
            em_key = em_form.strip().lower()
            if not em_key:
                continue
            target_em = lexicon_em.setdefault(em_key, [])
            if pt_word not in target_em:
                target_em.append(pt_word)

    # pronomes
    pronoun_map = _extract_pronoun_pt_map(pronouns)
    for norm_pt, pron_payload in pronoun_map.items():
        em_forms = pron_payload.get("forms", []) if isinstance(pron_payload, dict) else []
        pt_display = str(pron_payload.get("pt") or norm_pt) if isinstance(pron_payload, dict) else norm_pt
        if not em_forms:
            continue

        # PT -> Emakua
        pronoun_pt[norm_pt] = em_forms
        if norm_pt not in spell_vocab_pt:
            spell_vocab_pt[norm_pt] = pt_display

        # Emakua -> PT
        for em_form in em_forms:
            em_key = em_form.strip().lower()
            if not em_key:
                continue
            target_em = pronoun_em.setdefault(em_key, [])
            if pt_display not in target_em:
                target_em.append(pt_display)

    return lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em, grammar_profile


# --- Corretor ortogrÃ¡fico leve ---


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
    """Corretor ortogrÃ¡fico leve sÃ³ para portuguÃªs."""

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


def _find_similar_pt_keys(
    norm: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    max_items: int = 4,
) -> List[str]:
    if not norm:
        return []

    keys = set(lexicon_pt.keys()) | set(pronoun_pt.keys())
    out: List[Tuple[int, str]] = []

    for cand in keys:
        if not cand or cand == norm:
            continue

        # Similaridade conservadora para evitar "embaralhar" traducoes.
        dist = _levenshtein(norm, cand)
        max_dist = 1 if len(norm) <= 6 else 2
        if dist > max_dist:
            continue

        # Se distancia for maior (2), exige inicio parecido para manter contexto.
        if dist == 2:
            if len(norm) < 3 or len(cand) < 3 or norm[:3] != cand[:3]:
                continue

        out.append((dist, cand))

    out.sort(key=lambda item: (item[0], item[1]))
    return [cand for _, cand in out[:max_items]]


def lookup_pt_to_em(
    word: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    grammar_profile: Optional[Dict[str, Any]] = None,
    missing_log: Optional[List[str]] = None,
) -> Dict:
    """Lookup PT->Emakua com corretor leve em PT e ranking estavel."""
    corrected = correct_spelling_pt(word, spell_vocab_pt)
    norm = _normalize_pt(corrected)
    em_candidates: List[str] = []
    if norm in pronoun_pt:
        em_candidates.extend(pronoun_pt[norm])
    if norm in lexicon_pt:
        for v in lexicon_pt[norm]:
            if v not in em_candidates:
                em_candidates.append(v)

    # Regra principal: so busca semelhantes se NAO houver match exato.
    if not em_candidates:
        for similar_key in _find_similar_pt_keys(norm, lexicon_pt, pronoun_pt):
            for v in pronoun_pt.get(similar_key, []):
                if v not in em_candidates:
                    em_candidates.append(v)
            for v in lexicon_pt.get(similar_key, []):
                if v not in em_candidates:
                    em_candidates.append(v)
    ranked = _rank_candidates_for_emakua(em_candidates, grammar_profile or {})
    if not ranked and norm in _POSSESSIVE_SUFFIX_BY_PRONOUN:
        ranked = [_POSSESSIVE_SUFFIX_BY_PRONOUN[norm]]
    if len(ranked) > 4:
        ranked = ranked[:4]
    found = bool(ranked)
    if not found and missing_log is not None:
        missing_log.append(word)
    return {
        "source": word,
        "normalized": norm,
        "candidates": ranked,
        "found": found,
    }

def _resolve_preferred_pt_variant(
    token: str,
    preferred: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
) -> Optional[str]:
    """Finds a preferred PT->Emakhuwa variant even when it is outside top-N candidates."""

    pref = (preferred or "").strip()
    if not pref:
        return None

    corrected = correct_spelling_pt(token, spell_vocab_pt)
    norm = _normalize_pt(corrected)
    pref_norm = pref.lower()

    for cand in pronoun_pt.get(norm, []):
        if isinstance(cand, str) and cand.strip().lower() == pref_norm:
            return cand.strip()
    for cand in lexicon_pt.get(norm, []):
        if isinstance(cand, str) and cand.strip().lower() == pref_norm:
            return cand.strip()
    return None


def lookup_em_to_pt(
    word: str,
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Optional[Dict[str, Any]] = None,
    missing_log: Optional[List[str]] = None,
) -> Dict:
    """Lookup Emakua->PT com ordenacao estavel de variantes."""
    key = word.strip().lower()
    pt_candidates: List[str] = []
    if key in pronoun_em:
        pt_candidates.extend(pronoun_em[key])
    if key in lexicon_em:
        for v in lexicon_em[key]:
            if v not in pt_candidates:
                pt_candidates.append(v)
    ranked = sorted(_dedupe_keep_order(pt_candidates), key=lambda x: x.lower())
    if len(ranked) > 4:
        ranked = ranked[:4]
    found = bool(ranked)
    if not found and missing_log is not None:
        missing_log.append(word)
    return {
        "source": word,
        "normalized": key,
        "candidates": ranked,
        "found": found,
    }

# --- TokenizaÃ§Ã£o e construÃ§Ã£o de frase ---


def _tokenize(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r"([.,!?;:])", r" \1 ", text)
    return [t for t in text.split() if t]


def _rebuild_sentence_from_tokens(tokens: List[str]) -> str:
    sentence = " ".join(tokens)
    return re.sub(r"\s+([.,!?;:])", r"\1", sentence)


def _title_case_word(word: str) -> str:
    if not word:
        return word
    return word[:1].upper() + word[1:].lower()


def _normalize_list_case(words: List[str]) -> List[str]:
    return [_title_case_word(w.strip()) for w in words if w and w.strip()]


def _normalize_sentence_case(text: str) -> str:
    tokens = _tokenize(text)
    out: List[str] = []
    seen_word = False
    for tok in tokens:
        if _is_punctuation(tok):
            out.append(tok)
            continue
        if not seen_word:
            out.append(_title_case_word(tok))
            seen_word = True
        else:
            out.append(tok.lower())
    return _rebuild_sentence_from_tokens(out)


def _has_terminal_punctuation(text: str) -> bool:
    return bool(re.search(r"[.!?]\s*$", (text or "").strip()))


def _infer_terminal_punctuation(source_text: str, word_tokens: List[str]) -> str:
    normalized = [_normalize_pt(t) for t in word_tokens if t and not _is_punctuation(t)]
    if not normalized:
        return "."

    first = normalized[0]
    if first in _PT_QUESTION_HINTS:
        return "?"
    if len(normalized) >= 2 and normalized[0] == "por" and normalized[1] == "que":
        return "?"

    if first in _PT_EXCLAMATION_HINTS:
        return "!"
    if any(tok in _PT_EXCLAMATION_HINTS for tok in normalized):
        return "!"

    # fallback padrao
    return "."


def _apply_phrase_punctuation_policy(source_text: str, translated: str) -> str:
    out = (translated or "").strip()
    if not out:
        return out

    source_tokens = _tokenize(source_text or "")
    source_words = [t for t in source_tokens if not _is_punctuation(t)]

    # Regra solicitada: frases compostas (>2 palavras) nao devem conter virgulas.
    if len(source_words) > 2:
        out = out.replace(",", "")
        out = re.sub(r"\s{2,}", " ", out).strip()
        out = re.sub(r"\s+([.!?;:])", r"\1", out)

    # Se a frase de origem nao termina com pontuacao terminal, inferimos uma.
    # Em perguntas, adiciona '?' mesmo que o usuario nao tenha colocado.
    if len(source_words) > 1 and not _has_terminal_punctuation(source_text) and not _has_terminal_punctuation(out):
        out = f"{out}{_infer_terminal_punctuation(source_text, source_words)}"

    return out


def _reorder_quality_pattern_pt(tokens: List[str]) -> List[str]:
    """Regra 6: reordena adjetivo + substantivo para substantivo + adjetivo.

    Mantem pontuacao e cobre tanto frases de 2 palavras quanto frases maiores.
    """
    words_with_idx = [(idx, tok) for idx, tok in enumerate(tokens) if not _is_punctuation(tok)]
    if len(words_with_idx) < 2:
        return tokens

    result = list(tokens)
    for i in range(len(words_with_idx) - 1):
        idx_a, tok_a = words_with_idx[i]
        idx_b, tok_b = words_with_idx[i + 1]
        norm_a = _normalize_pt(tok_a)
        norm_b = _normalize_pt(tok_b)
        if norm_a in _PT_ADJECTIVE_HINTS and norm_b not in _PT_ADJECTIVE_HINTS:
            # Swap local pair to enforce noun + adjective.
            result[idx_a], result[idx_b] = result[idx_b], result[idx_a]
    return result


def _prepare_pt_tokens(tokens: List[str]) -> Tuple[List[str], Dict[int, str]]:
    """Aplica regras de prÃ©-processamento PT:
    - remove artigos definidos e 'da'
    - remove pronomes possessivos e marca sufixo para a prÃ³xima palavra de conteÃºdo
    """
    out_tokens: List[str] = []
    suffix_by_word_pos: Dict[int, str] = {}
    pending_suffix: Optional[str] = None
    pending_demo_suffix: Optional[str] = None
    word_pos = 0
    first_word_norm: Optional[str] = None

    filtered_word_norms: List[str] = []
    for tok in tokens:
        if _is_punctuation(tok):
            continue
        norm = _normalize_pt(tok)
        if first_word_norm is None:
            first_word_norm = norm
        if norm in _OMIT_PT_TOKENS:
            continue
        if norm in _POSSESSIVE_SUFFIX_BY_PRONOUN:
            continue
        if first_word_norm == norm and norm in _DEMONSTRATIVE_SUFFIX_BY_TOKEN:
            continue
        filtered_word_norms.append(norm)
    has_content_words = bool(filtered_word_norms)

    kept_word_index = 0
    for tok in tokens:
        if _is_punctuation(tok):
            out_tokens.append(tok)
            continue

        norm = _normalize_pt(tok)
        if norm in _OMIT_PT_TOKENS:
            continue

        if first_word_norm is None:
            first_word_norm = norm

        demo_suffix = _DEMONSTRATIVE_SUFFIX_BY_TOKEN.get(norm)
        if (
            demo_suffix is not None
            and first_word_norm == norm
            and word_pos == 0
            and pending_demo_suffix is None
        ):
            pending_demo_suffix = demo_suffix
            continue

        suffix = _POSSESSIVE_SUFFIX_BY_PRONOUN.get(norm)
        if suffix is not None:
            # Mantem pronome possessivo isolado para permitir traducao direta
            # (ex.: "meu" -> "ka"), mesmo sem entrada explicita no banco.
            if not has_content_words:
                out_tokens.append(tok)
                word_pos += 1
                kept_word_index += 1
                continue
            # Posposicao: "carro meu" -> sufixo no substantivo anterior.
            if word_pos > 0 and kept_word_index >= 1:
                prev_norm = filtered_word_norms[kept_word_index - 1]
                if (
                    prev_norm not in _PT_ADJECTIVE_HINTS
                    and prev_norm not in _CONNECTOR_PT_TOKENS
                ):
                    suffix_by_word_pos[word_pos] = suffix
                    pending_suffix = None
                    continue

            pending_suffix = suffix
            continue

        out_tokens.append(tok)
        word_pos += 1
        kept_word_index += 1

        if pending_demo_suffix:
            # Regra nova: "esse/essa" no inicio da frase aplica sufixo apenas
            # para pessoas/animais (nao objetos).
            if norm in _PT_ADJECTIVE_HINTS:
                continue
            if norm in _PT_PERSON_ANIMAL_HINTS:
                existing = suffix_by_word_pos.get(word_pos, "")
                if pending_demo_suffix not in existing:
                    suffix_by_word_pos[word_pos] = f"{existing}{pending_demo_suffix}"
                pending_demo_suffix = None
            else:
                # Proximo termo nao e pessoa/animal: nao aplica.
                pending_demo_suffix = None

        if pending_suffix:
            # Se vier adjetivo apos possessivo (ex.: "meu lindo carro"),
            # aguardamos o proximo substantivo para receber o sufixo.
            if norm in _PT_ADJECTIVE_HINTS:
                continue
            suffix_by_word_pos[word_pos] = pending_suffix
            pending_suffix = None

    # Fallback: se nao encontramos substantivo depois, aplica no ultimo termo de conteudo.
    if pending_suffix and word_pos > 0:
        suffix_by_word_pos[word_pos] = pending_suffix

    return out_tokens, suffix_by_word_pos


def _extract_possessive_list_parts_pt(text: str) -> Optional[List[str]]:
    """Detecta listas PT com possessivo + virgulas + conector final (e/e o/e a/e os/e as)."""
    source = re.sub(r"\s+", " ", (text or "").strip())
    if not source or "," not in source:
        return None

    norm_source = _normalize_pt(source)
    has_trigger_pronoun = any(
        re.search(rf"\b{pron}\b", norm_source)
        for pron in _POSSESSIVE_LIST_TRIGGER_PRONOUNS
    )
    if not has_trigger_pronoun:
        return None

    # Divide em "cabeca, penultimo + conector + ultimo".
    match = re.match(
        r"^(?P<head>.+),(?P<before_last>[^,]+?)\s+e(?:\s+(?:o|a|os|as))?\s+(?P<last>[^,]+)\s*$",
        source,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    head = str(match.group("head") or "").strip()
    before_last = str(match.group("before_last") or "").strip()
    last = str(match.group("last") or "").strip()
    if not head or not before_last or not last:
        return None

    head_items = [part.strip() for part in head.split(",") if part.strip()]
    if not head_items:
        return None

    return head_items + [before_last, last]


def _extract_possessive_comma_list_parts_pt(text: str) -> Optional[List[str]]:
    """Detecta lista possessiva separada por virgula, sem conector final."""
    source = re.sub(r"\s+", " ", (text or "").strip())
    if not source or "," not in source:
        return None

    # Se existe conector final, a regra de "ni" deve tratar.
    if re.search(r"\s+e(?:\s+(?:o|a|os|as))?\s+", source, flags=re.IGNORECASE):
        return None

    parts = [part.strip() for part in source.split(",") if part.strip()]
    if len(parts) < 2:
        return None

    # Ativa so quando todos os itens forem possessivos alvo.
    for item in parts:
        norm_item = _normalize_pt(item)
        has_pron = any(
            re.search(rf"\b{pron}\b", norm_item)
            for pron in _POSSESSIVE_LIST_TRIGGER_PRONOUNS
        )
        if not has_pron:
            return None

    return parts


def _strip_pt_copula(text: str) -> str:
    tokens = _tokenize(text or "")
    kept: List[str] = []
    for tok in tokens:
        if _is_punctuation(tok):
            kept.append(tok)
            continue
        if _normalize_pt(tok) in _COPULA_PT_TOKENS:
            continue
        kept.append(tok)
    return _rebuild_sentence_from_tokens(kept).strip()


def _is_quality_clause_pt(text: str) -> bool:
    tokens = [t for t in _tokenize(text or "") if not _is_punctuation(t)]
    if len(tokens) < 2:
        return False
    normalized = [_normalize_pt(t) for t in tokens]
    has_copula = any(tok in _COPULA_PT_TOKENS for tok in normalized)
    if not has_copula:
        return False
    # Heuristica simples: ultima palavra da clausula costuma ser adjetivo.
    if normalized[-1] in _PT_ADJECTIVE_HINTS:
        return True
    return False


def _copula_translation_for_context(prev_norm: str) -> str:
    if prev_norm in _PT_LIVING_HINTS:
        return "to"
    return "tiyo"


def _extract_quality_pair_parts_pt(text: str) -> Optional[Tuple[str, str]]:
    source = re.sub(r"\s+", " ", (text or "").strip())
    if not source:
        return None

    # Conector de juncao entre clausulas: e / e o / e a / e os / e as
    connector_match = None
    for m in re.finditer(r"\s+e(?:\s+(?:o|a|os|as))?\s+", source, flags=re.IGNORECASE):
        left = source[: m.start()].strip()
        right = source[m.end() :].strip()
        if not left or not right:
            continue
        if _is_quality_clause_pt(left) and _is_quality_clause_pt(right):
            connector_match = (left, right)
            break
    return connector_match


def _canonicalize_phrase_pt(text: str) -> str:
    tokens = _tokenize(text)
    out: List[str] = []
    for tok in tokens:
        if _is_punctuation(tok):
            out.append(tok)
        else:
            out.append(_normalize_pt(tok))
    phrase = " ".join(out)
    return re.sub(r"\s+([.,!?;:])", r"\1", phrase)


def _translate_pt_to_em_with_indexes(
    text: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Dict[str, Any],
    apply_phrase_policy: bool = True,
) -> str:
    tokens = _tokenize(text)
    tokens = _reorder_quality_pattern_pt(tokens)
    tokens, possessive_suffix_by_position = _prepare_pt_tokens(tokens)
    word_tokens = [t for t in tokens if not _is_punctuation(t)]
    phrase_preferences: Dict[int, str] = {}
    if len(word_tokens) > 1:
        canonical = _canonicalize_phrase_pt(text)
        rows = get_phrase_memory_preferences(canonical)
        learned_words = [
            str((row or {}).get("selected_macua") or "").strip()
            for row in rows
        ]
        if len(learned_words) == len(word_tokens) and all(learned_words):
            sentence = _build_sentence_from_memory_words(tokens, learned_words)
            return (
                _apply_phrase_punctuation_policy(text, sentence)
                if apply_phrase_policy
                else sentence
            )
        for row in rows:
            try:
                pos = int(row.get("position_index") or 0)
            except Exception:
                pos = 0
            selected = str(row.get("selected_macua") or "").strip()
            if pos > 0 and selected:
                phrase_preferences[pos] = selected

    sentence = _build_sentence_from_lookup(
        tokens,
        "pt_to_em",
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
        phrase_pt_preferences=phrase_preferences,
        possessive_suffix_by_position=possessive_suffix_by_position,
    )
    if apply_phrase_policy:
        return _apply_phrase_punctuation_policy(text, sentence)
    return sentence


def _translate_possessive_list_with_ni(
    text: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Dict[str, Any],
) -> Optional[str]:
    parts = _extract_possessive_list_parts_pt(text)
    if not parts:
        return None

    translated_parts: List[str] = []
    for part in parts:
        translated = _translate_pt_to_em_with_indexes(
            part,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
            apply_phrase_policy=False,
        ).strip()
        cleaned = re.sub(r"\s+([.,!?;:])", r"\1", translated)
        cleaned = cleaned.strip(" \t\r\n.,!?;:")
        if not cleaned:
            return None
        translated_parts.append(cleaned.lower())

    if len(translated_parts) < 2:
        return None
    if len(translated_parts) == 2:
        return f"{translated_parts[0]} ni {translated_parts[1]}"
    return f"{', '.join(translated_parts[:-1])} ni {translated_parts[-1]}"


def _translate_possessive_comma_list(
    text: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Dict[str, Any],
) -> Optional[str]:
    parts = _extract_possessive_comma_list_parts_pt(text)
    if not parts:
        return None

    translated_parts: List[str] = []
    for part in parts:
        translated = _translate_pt_to_em_with_indexes(
            part,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
            apply_phrase_policy=False,
        ).strip()
        cleaned = re.sub(r"\s+([.,!?;:])", r"\1", translated)
        cleaned = cleaned.strip(" \t\r\n.,!?;:")
        if not cleaned:
            return None
        translated_parts.append(cleaned.lower())

    if len(translated_parts) < 2:
        return None
    return ", ".join(translated_parts)


def _translate_quality_pair_with_ni(
    text: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Dict[str, Any],
) -> Optional[str]:
    pair = _extract_quality_pair_parts_pt(text)
    if not pair:
        return None
    left_raw, right_raw = pair
    # Mantem copula dentro de cada clausula para permitir regra "e/é -> tiyo"
    # quando estiver caracterizando substantivo.
    left = left_raw
    right = right_raw
    if not left or not right:
        return None

    left_t = _translate_pt_to_em_with_indexes(
        left,
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
        apply_phrase_policy=False,
    ).strip()
    right_t = _translate_pt_to_em_with_indexes(
        right,
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
        apply_phrase_policy=False,
    ).strip()
    if not left_t or not right_t:
        return None

    left_clean = re.sub(r"\s+([.,!?;:])", r"\1", left_t).strip(" \t\r\n.,!?;:").lower()
    right_clean = re.sub(r"\s+([.,!?;:])", r"\1", right_t).strip(" \t\r\n.,!?;:").lower()
    if not left_clean or not right_clean:
        return None
    combined = f"{left_clean} ni {right_clean}".strip()
    if not combined:
        return None
    return combined[:1].upper() + combined[1:]


def _build_sentence_from_memory_words(tokens: List[str], selected_words: List[str]) -> str:
    """Builds sentence preserving punctuation slots, using learned word order."""

    out_tokens: List[str] = []
    idx = 0
    for tok in tokens:
        if _is_punctuation(tok):
            out_tokens.append(tok)
            continue
        replacement = selected_words[idx] if idx < len(selected_words) else tok
        out_tokens.append(replacement)
        idx += 1

    sentence = " ".join(out_tokens)
    sentence = re.sub(r"\s+([.,!?;:])", r"\1", sentence)
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]
    return sentence


def _build_sentence_from_lookup(
    tokens: List[str],
    direction: str,
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    spell_vocab_pt: Dict[str, str],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
    grammar_profile: Dict[str, Any],
    phrase_pt_preferences: Optional[Dict[int, str]] = None,
    possessive_suffix_by_position: Optional[Dict[int, str]] = None,
) -> str:
    missing: List[str] = []
    out_tokens: List[str] = []
    # Caso especial: entrada de Ãºnica palavra (sem pontuaÃ§Ã£o)
    # Retorna atÃ© 4 traduÃ§Ãµes possÃ­veis da palavra.
    if len(tokens) == 1 and not _is_punctuation(tokens[0]):
        tok = tokens[0]
        if direction == "pt_to_em":
            info = lookup_pt_to_em(
                tok,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                grammar_profile=grammar_profile,
                missing_log=missing,
            )
        else:  # em_to_pt
            info = lookup_em_to_pt(
                tok,
                lexicon_em,
                pronoun_em,
                grammar_profile=grammar_profile,
                missing_log=missing,
            )

        candidates = info["candidates"][:4]
        if candidates:
            if direction == "pt_to_em" and possessive_suffix_by_position:
                # Regra de possessivo tambem para casos que viram
                # "frase de 1 palavra" apos preprocessamento (ex.: "Minha casa").
                chosen = candidates[0]
                suffix = possessive_suffix_by_position.get(1)
                if suffix and not chosen.lower().endswith(suffix.lower()):
                    chosen = f"{chosen}{suffix}"
                return _normalize_sentence_case(chosen)
            # Regra 1 (lista): cada item inicia com maiÃºscula.
            sentence = ", ".join(_normalize_list_case(candidates))
            return sentence

        # Sem candidatos conhecidos, devolve a prÃ³pria palavra
        return tok
    word_position = 0
    for idx, tok in enumerate(tokens):
        if _is_punctuation(tok):
            out_tokens.append(tok)
            continue

        if direction == "pt_to_em" and _normalize_pt(tok) == "e" and len(tokens) > 1:
            # Em padrao de elogio/atribuicao ("X e bonito"), nao gerar "ni".
            # Em mencao/lista ("carro e galinha"), mantemos o conector.
            prev_norm = ""
            for j in range(idx - 1, -1, -1):
                if _is_punctuation(tokens[j]):
                    continue
                prev_norm = _normalize_pt(tokens[j])
                break
            next_norm = ""
            for j in range(idx + 1, len(tokens)):
                if _is_punctuation(tokens[j]):
                    continue
                next_norm = _normalize_pt(tokens[j])
                break
            if next_norm in _PT_ADJECTIVE_HINTS:
                out_tokens.append(_copula_translation_for_context(prev_norm))
                continue

        word_position += 1
        if direction == "pt_to_em":
            info = lookup_pt_to_em(
                tok,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                grammar_profile=grammar_profile,
                missing_log=missing,
            )
        else:  # em_to_pt
            info = lookup_em_to_pt(
                tok,
                lexicon_em,
                pronoun_em,
                grammar_profile=grammar_profile,
                missing_log=missing,
            )

        candidates = info["candidates"]
        if candidates:
            if direction == "pt_to_em" and phrase_pt_preferences:
                preferred = phrase_pt_preferences.get(word_position)
                if preferred:
                    selected = next(
                        (c for c in candidates if c.lower() == preferred.lower()),
                        None,
                    )
                    if not selected:
                        selected = _resolve_preferred_pt_variant(
                            tok,
                            preferred,
                            lexicon_pt,
                            pronoun_pt,
                            spell_vocab_pt,
                        )
                    chosen = selected or candidates[0]
                else:
                    chosen = candidates[0]
            else:
                chosen = candidates[0]

            if direction == "pt_to_em" and possessive_suffix_by_position:
                suffix = possessive_suffix_by_position.get(word_position)
                if suffix and not chosen.lower().endswith(suffix.lower()):
                    chosen = f"{chosen}{suffix}"
            out_tokens.append(chosen)
        else:
            out_tokens.append(tok)

    sentence = _rebuild_sentence_from_tokens(out_tokens)
    # Regra 1 (frase): sÃ³ a primeira palavra em maiÃºscula.
    return _normalize_sentence_case(sentence)


def _count_known_tokens(
    tokens: List[str],
    lexicon_pt: Dict[str, List[str]],
    pronoun_pt: Dict[str, List[str]],
    lexicon_em: Dict[str, List[str]],
    pronoun_em: Dict[str, List[str]],
) -> Tuple[int, int]:
    """Conta quantos tokens parecem PT e quantos parecem Emakua.

    Usa apenas presenÃ§a nos Ã­ndices (sem correÃ§Ã£o),
    para nÃ£o distorcer a detecÃ§Ã£o de lÃ­ngua.
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
    """Detecta automaticamente se a frase Ã© PT ou Emakua."""

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
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em, grammar_profile = _build_indexes(resources)

    try:
        quality_pair = _translate_quality_pair_with_ni(
            text,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
        )
    except Exception:
        quality_pair = None
    if quality_pair is not None:
        return _apply_phrase_punctuation_policy(text, quality_pair)

    try:
        special_list = _translate_possessive_list_with_ni(
            text,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
        )
    except Exception:
        special_list = None
    if special_list is not None:
        return special_list
    try:
        special_comma_list = _translate_possessive_comma_list(
            text,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
        )
    except Exception:
        special_comma_list = None
    if special_comma_list is not None:
        return special_comma_list

    return _translate_pt_to_em_with_indexes(
        text,
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
        apply_phrase_policy=True,
    )


def translate_em_to_pt(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em, grammar_profile = _build_indexes(resources)

    tokens = _tokenize(text)
    sentence = _build_sentence_from_lookup(
        tokens,
        "em_to_pt",
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
    )
    return _apply_phrase_punctuation_policy(text, sentence)


def translate(text: str, direction: str = "auto") -> str:
    """TraduÃ§Ã£o de texto entre PT e Emakua.

    direction:
      - "pt_to_em": forÃ§a PT -> Emakua
      - "em_to_pt": forÃ§a Emakua -> PT
      - "auto"   : detecta automaticamente
    """

    text = text.strip()
    if not text:
        return ""

    # Carrega recursos dinamicamente (com cache TTL) a cada chamada,
    # garantindo que a consulta ao Supabase faÃ§a parte do fluxo da rota.
    resources = load_resources()
    lexicon_pt, pronoun_pt, spell_vocab_pt, lexicon_em, pronoun_em, grammar_profile = _build_indexes(resources)

    tokens = _tokenize(text)

    if direction == "pt_to_em":
        try:
            quality_pair = _translate_quality_pair_with_ni(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            quality_pair = None
        if quality_pair is not None:
            return _apply_phrase_punctuation_policy(text, quality_pair)
        try:
            special_list = _translate_possessive_list_with_ni(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            special_list = None
        if special_list is not None:
            return special_list
        try:
            special_comma_list = _translate_possessive_comma_list(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            special_comma_list = None
        if special_comma_list is not None:
            return special_comma_list
        return _translate_pt_to_em_with_indexes(
            text,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
            apply_phrase_policy=True,
        )
    if direction == "em_to_pt":
        sentence = _build_sentence_from_lookup(
            tokens,
            "em_to_pt",
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
        )
        return _apply_phrase_punctuation_policy(text, sentence)

    auto_dir = _detect_direction(tokens, lexicon_pt, pronoun_pt, lexicon_em, pronoun_em)
    phrase_preferences: Dict[int, str] = {}
    possessive_suffix_by_position: Dict[int, str] = {}
    if auto_dir == "pt_to_em":
        try:
            quality_pair = _translate_quality_pair_with_ni(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            quality_pair = None
        if quality_pair is not None:
            return _apply_phrase_punctuation_policy(text, quality_pair)
        try:
            special_list = _translate_possessive_list_with_ni(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            special_list = None
        if special_list is not None:
            return special_list
        try:
            special_comma_list = _translate_possessive_comma_list(
                text,
                lexicon_pt,
                pronoun_pt,
                spell_vocab_pt,
                lexicon_em,
                pronoun_em,
                grammar_profile,
            )
        except Exception:
            special_comma_list = None
        if special_comma_list is not None:
            return special_comma_list
        return _translate_pt_to_em_with_indexes(
            text,
            lexicon_pt,
            pronoun_pt,
            spell_vocab_pt,
            lexicon_em,
            pronoun_em,
            grammar_profile,
            apply_phrase_policy=True,
        )

    sentence = _build_sentence_from_lookup(
        tokens,
        auto_dir,
        lexicon_pt,
        pronoun_pt,
        spell_vocab_pt,
        lexicon_em,
        pronoun_em,
        grammar_profile,
        phrase_pt_preferences=phrase_preferences,
        possessive_suffix_by_position=possessive_suffix_by_position,
    )
    return _apply_phrase_punctuation_policy(text, sentence)


