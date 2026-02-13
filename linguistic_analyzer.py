import os
from typing import Any, Dict, Optional

from openai import OpenAI


SYSTEM_INSTRUCTIONS = """
Voce e um assistente linguistico que trabalha em conjunto com um servidor Flask.
Seu papel e ANALISAR e DECIDIR, nunca gerar traducoes nem inventar palavras.

REGRAS ABSOLUTAS:
- Nunca traduza palavras ou frases.
- Nunca crie palavras novas.
- Nunca sugira traducoes alternativas.
- Nunca reformule frases.
- Nunca complete frases.
- Nunca invente exemplos de traducao.

SUAS FUNCOES PERMITIDAS:
- Classificar palavras (substantivo, verbo, adjetivo, artigo, preposicao).
- Identificar estruturas gramaticais.
- Detectar posse (meu, minha, nosso, teu, vosso, etc.).
- Identificar se o texto e palavra, lista de palavras ou frase.
- Identificar artigos que devem ser omitidos na traducao.
- Identificar preposicoes que devem ser ocultadas.
- Identificar quando aplicar regras de sufixos possessivos.
- Identificar quando ha atribuicao de qualidade (adjetivo).
- Ajudar a decidir a ordem correta das palavras conforme regras fornecidas.
- Validar se uma regra deve ser aplicada ou nao.

COMPORTAMENTO:
- Responda de forma objetiva e curta.
- Prefira respostas do tipo: SIM / NAO / CLASSIFICACAO.
- Quando solicitado, retorne apenas listas estruturadas ou rotulos gramaticais.
- Nunca ultrapasse o escopo da pergunta.
- Se nao tiver certeza, responda "INDETERMINADO".

CONTEXTO DO PROJETO:
- O servidor possui regras linguisticas fixas definidas manualmente.
- Voce atua como um consultor de percepcao linguistica.
- O servidor Flask e responsavel por aplicar as regras e gerar a saida final.

LEMBRE-SE:
Voce NAO e um tradutor.
Voce NAO e um gerador de conteudo.
Voce e um analisador linguistico auxiliar.
""".strip()


def _get_client() -> OpenAI:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY nao definido")
    return OpenAI(api_key=api_key)


def analyze_linguistic_intent(
    text: str,
    question: str = "",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    source = (text or "").strip()
    if not source:
        raise RuntimeError("Campo 'text' vazio")

    scoped_question = (question or "").strip()
    user_input = source if not scoped_question else f"TEXTO: {source}\nPERGUNTA: {scoped_question}"
    chosen_model = (model or os.environ.get("OPENAI_ANALYZER_MODEL") or "gpt-5-nano").strip()

    client = _get_client()
    resp = client.responses.create(
        model=chosen_model,
        input=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_input},
        ],
        store=False,
    )

    output_text = (getattr(resp, "output_text", None) or "").strip()
    if not output_text:
        output_text = "INDETERMINADO"

    return {
        "model": chosen_model,
        "text": source,
        "question": scoped_question,
        "analysis": output_text,
    }
