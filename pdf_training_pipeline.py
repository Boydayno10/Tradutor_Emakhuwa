import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from pypdf import PdfReader


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DIR = BASE_DIR / "Arquivos de treinamento"
OUTPUT_DIR = TRAINING_DIR / "processed"
CORPUS_FILE = OUTPUT_DIR / "training_corpus.txt"
KNOWLEDGE_FILE = OUTPUT_DIR / "training_knowledge.json"


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _strip_accents(text: str) -> str:
    value = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def _tokenize_words(text: str) -> List[str]:
    clean = _strip_accents(text.lower())
    return re.findall(r"[a-z]+(?:'[a-z]+)?", clean)


def _extract_pdf_text(pdf_path: Path) -> Tuple[str, int]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        clean = _normalize_spaces(raw)
        if clean:
            pages.append(clean)
    return "\n".join(pages), len(reader.pages)


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    raw_parts = re.split(r"(?<=[.!?])\s+", text)
    out: List[str] = []
    for part in raw_parts:
        line = _normalize_spaces(part)
        if len(line) < 3:
            continue
        out.append(line)
    # dedupe preserving order
    seen = set()
    deduped = []
    for sentence in out:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(sentence)
    return deduped


def _build_rule_markers(tokens_freq: Counter) -> Dict[str, List[str]]:
    # Marcadores para ajudar o classificador linguistico (sem criar traducoes).
    seed_groups = {
        "possessive_pronouns": [
            "meu",
            "minha",
            "meus",
            "minhas",
            "nosso",
            "nossa",
            "seu",
            "sua",
            "teu",
            "tua",
            "vosso",
            "vossa",
        ],
        "articles": ["o", "a", "os", "as"],
        "prepositions": ["da", "de", "do", "dos", "das", "em", "no", "na"],
        "question_markers": ["quem", "que", "qual", "quais", "quando", "onde", "como", "porque"],
        "quality_markers": ["bonito", "bonita", "caro", "cara", "grande", "pequeno", "novo", "velho"],
    }

    result: Dict[str, List[str]] = {}
    for group, candidates in seed_groups.items():
        present = [w for w in candidates if tokens_freq[w] > 0]
        result[group] = present
    return result


def build_training_artifacts() -> Dict:
    if not TRAINING_DIR.exists():
        raise RuntimeError(f"Pasta de treino nao encontrada: {TRAINING_DIR}")

    pdf_files = sorted(TRAINING_DIR.glob("*.pdf"))
    if not pdf_files:
        raise RuntimeError(f"Nenhum PDF encontrado em: {TRAINING_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_text_parts: List[str] = []
    pdf_stats: List[Dict] = []
    for pdf in pdf_files:
        text, page_count = _extract_pdf_text(pdf)
        all_text_parts.append(text)
        pdf_stats.append(
            {
                "file": pdf.name,
                "pages": page_count,
                "chars": len(text),
            }
        )

    merged_text = "\n".join(part for part in all_text_parts if part)
    sentences = _split_sentences(merged_text)
    tokens = _tokenize_words(merged_text)
    token_freq = Counter(tokens)

    top_tokens = [{"token": t, "count": c} for t, c in token_freq.most_common(300)]
    rule_markers = _build_rule_markers(token_freq)

    CORPUS_FILE.write_text("\n".join(sentences), encoding="utf-8")

    knowledge = {
        "source": "pdf_training_pipeline",
        "source_dir": str(TRAINING_DIR),
        "pdfs": pdf_stats,
        "sentence_count": len(sentences),
        "token_count": len(tokens),
        "unique_token_count": len(token_freq),
        "top_tokens": top_tokens,
        "rule_markers": rule_markers,
    }
    KNOWLEDGE_FILE.write_text(
        json.dumps(knowledge, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return knowledge


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai PDFs de treinamento e gera corpus/knowledge para ML linguistico."
    )
    parser.parse_args()
    knowledge = build_training_artifacts()
    print(
        json.dumps(
            {
                "status": "ok",
                "sentence_count": knowledge["sentence_count"],
                "token_count": knowledge["token_count"],
                "output_corpus": str(CORPUS_FILE),
                "output_knowledge": str(KNOWLEDGE_FILE),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

