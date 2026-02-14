import os

from flask import Flask, jsonify, request

from correction_service import (
    delete_entry,
    delete_variant,
    get_correction_payload,
    get_phrase_correction_payload,
    save_phrase_learning,
    upsert_variants,
)
from linguistic_analyzer import analyze_linguistic_intent
from participar_service import get_participar_config
from pdf_training_pipeline import build_training_artifacts
from supabase_client_strict import get_client
from training_knowledge_loader import clear_training_knowledge_cache, load_training_knowledge
from translation_pipeline import translate
from vote_service import VoteCooldownError, register_translation_vote

app = Flask(__name__)


@app.after_request
def add_no_cache_headers(response):
    """Guarantees no HTTP response is cached by clients/proxies."""

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.route("/api/<path:_path>", methods=["OPTIONS"])
def options_api(_path: str):
    return ("", 204)


def _get_bearer_token() -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _require_authorized_profile():
    token = _get_bearer_token()
    if not token:
        return None, (jsonify({"error": "Token ausente"}), 401)

    client = get_client()
    try:
        user_resp = client.auth.get_user(token)
        user = getattr(user_resp, "user", None)
        user_id = getattr(user, "id", None) if user is not None else None
    except Exception as exc:
        return None, (jsonify({"error": f"Falha ao validar token: {exc}"}), 401)

    if not user_id:
        return None, (jsonify({"error": "Sessao invalida"}), 401)

    try:
        profile_resp = (
            client.table("profiles")
            .select("id,authorized")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        rows = getattr(profile_resp, "data", None) or []
    except Exception as exc:
        return None, (jsonify({"error": f"Falha ao verificar perfil: {exc}"}), 500)

    if not rows or rows[0].get("authorized") is not True:
        return None, (jsonify({"error": "Utilizador sem permissao para correcao"}), 403)
    return rows[0], None


def _require_authenticated_user():
    token = _get_bearer_token()
    if not token:
        return None, (jsonify({"error": "Token ausente"}), 401)

    client = get_client()
    try:
        user_resp = client.auth.get_user(token)
        user = getattr(user_resp, "user", None)
        user_id = getattr(user, "id", None) if user is not None else None
    except Exception as exc:
        return None, (jsonify({"error": f"Falha ao validar token: {exc}"}), 401)

    if not user_id:
        return None, (jsonify({"error": "Sessao invalida"}), 401)
    return {"user_id": user_id}, None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/translate", methods=["POST"])
def translate_route():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    direction = data.get("direction", "auto")  # auto, pt_to_em or em_to_pt

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Campo 'text' e obrigatorio"}), 400

    if direction not in {"auto", "pt_to_em", "em_to_pt"}:
        return jsonify({"error": "direction invalido"}), 400

    try:
        output = translate(text, direction=direction)
        return jsonify({"text": text, "direction": direction, "translation": output})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao", methods=["GET"])
def correcao_get_route():
    palavra = (request.args.get("palavra") or "").strip()
    if not palavra:
        return jsonify({"error": "Parametro 'palavra' e obrigatorio"}), 400
    try:
        payload = get_correction_payload(palavra)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/feedback/vote", methods=["POST"])
def feedback_vote_route():
    auth_payload, auth_error = _require_authenticated_user()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}
    source_text = str(data.get("source_text") or "").strip()
    translated_text = str(data.get("translated_text") or "").strip()
    direction = str(data.get("direction") or "").strip()
    vote_raw = data.get("vote")
    try:
        vote = int(vote_raw)
    except Exception:
        vote = 0

    if not source_text:
        return jsonify({"error": "Campo 'source_text' e obrigatorio"}), 400
    if not translated_text:
        return jsonify({"error": "Campo 'translated_text' e obrigatorio"}), 400
    if vote not in (-1, 1):
        return jsonify({"error": "Campo 'vote' deve ser -1 ou 1"}), 400

    # Votacao impacta ranking PT->Emakhuwa.
    if direction and direction != "pt_to_em":
        return jsonify({"error": "Votacao disponivel apenas para pt_to_em"}), 400

    try:
        payload = register_translation_vote(
            user_id=str(auth_payload.get("user_id") or ""),
            source_text=source_text,
            translated_text=translated_text,
            vote=vote,
        )
        return jsonify(payload)
    except VoteCooldownError as exc:
        return (
            jsonify(
                {
                    "error": str(exc),
                    "retry_after_seconds": int(exc.retry_after_seconds),
                    "next_allowed_at": exc.next_allowed_at,
                }
            ),
            429,
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/participar/config", methods=["GET"])
def participar_config_route():
    try:
        payload = get_participar_config(max_per_topic=3)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/linguistic/analyze", methods=["POST"])
def linguistic_analyze_route():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text") or "").strip()
    question = str(data.get("question") or "").strip()
    model = str(data.get("model") or "").strip()

    if not text:
        return jsonify({"error": "Campo 'text' e obrigatorio"}), 400

    try:
        payload = analyze_linguistic_intent(
            text=text,
            question=question,
            model=model or None,
        )
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/training/status", methods=["GET"])
def training_status_route():
    data = load_training_knowledge()
    if not data:
        return jsonify({"status": "empty", "message": "Treino PDF ainda nao gerado"}), 200
    return jsonify(
        {
            "status": "ok",
            "sentence_count": data.get("sentence_count", 0),
            "token_count": data.get("token_count", 0),
            "unique_token_count": data.get("unique_token_count", 0),
            "pdfs": data.get("pdfs", []),
        }
    )


@app.route("/api/training/rebuild", methods=["POST"])
def training_rebuild_route():
    _, auth_error = _require_authorized_profile()
    if auth_error:
        return auth_error
    try:
        payload = build_training_artifacts()
        clear_training_knowledge_cache()
        return jsonify(
            {
                "status": "ok",
                "sentence_count": payload.get("sentence_count", 0),
                "token_count": payload.get("token_count", 0),
                "unique_token_count": payload.get("unique_token_count", 0),
                "pdfs": payload.get("pdfs", []),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao/frase", methods=["GET"])
def correcao_phrase_get_route():
    texto = (request.args.get("texto") or "").strip()
    if not texto:
        return jsonify({"error": "Parametro 'texto' e obrigatorio"}), 400
    try:
        payload = get_phrase_correction_payload(texto)
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao/salvar", methods=["POST"])
def correcao_save_route():
    _, auth_error = _require_authorized_profile()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}
    variantes = data.get("variantes")
    if not isinstance(variantes, list):
        return jsonify({"error": "Campo 'variantes' deve ser uma lista"}), 400

    if not variantes:
        return jsonify({"error": "Envie pelo menos uma variante"}), 400

    try:
        result = upsert_variants(variantes)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao/frase/salvar", methods=["POST"])
def correcao_phrase_save_route():
    _, auth_error = _require_authorized_profile()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}
    try:
        result = save_phrase_learning(data)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao/apagar", methods=["POST"])
def correcao_delete_variant_route():
    _, auth_error = _require_authorized_profile()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}
    pt = str(data.get("pt") or "").strip()
    macua = str(data.get("macua") or "").strip()
    if not pt or not macua:
        return jsonify({"error": "Campos 'pt' e 'macua' sao obrigatorios"}), 400

    try:
        deleted = delete_variant(pt, macua)
        return jsonify({"deleted": deleted})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/correcao/remover_entrada", methods=["POST"])
def correcao_delete_entry_route():
    _, auth_error = _require_authorized_profile()
    if auth_error:
        return auth_error

    data = request.get_json(silent=True) or {}
    pt = str(data.get("pt") or "").strip()
    if not pt:
        return jsonify({"error": "Campo 'pt' e obrigatorio"}), 400

    try:
        deleted = delete_entry(pt)
        return jsonify({"deleted": deleted})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
