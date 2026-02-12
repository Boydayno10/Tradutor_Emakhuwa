import json
import sys
import urllib.request

API_URL = "http://localhost:5000/translate"


def call_api(text: str) -> str:
    data = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode("utf-8")
    payload = json.loads(body)
    if "translation" not in payload:
        raise RuntimeError(f"Resposta inesperada da API: {payload}")
    return payload["translation"]


def main(argv: list[str]) -> None:
    if len(argv) >= 2:
        text = " ".join(argv[1:])
    else:
        text = input("Texto em português: ").strip()
    if not text:
        print("")
        return
    try:
        translation = call_api(text)
        print(translation)
    except Exception as exc:  # noqa: BLE001
        print(f"Erro: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)
