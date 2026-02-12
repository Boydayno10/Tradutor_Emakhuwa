import os

from flask import Flask, jsonify, request

from correction_service import delete_entry, delete_variant, get_correction_payload, upsert_variants
from supabase_client_strict import get_client
from translation_pipeline import translate

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
