from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from src.config import DATA_RAW_DIR
from src.llm import init_llm
from src.rag import DocumentNotLoadedError, process_document, process_prompt

app = Flask(__name__)
CORS(app)

init_llm()


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/ingest")
def ingest():
    if "file" not in request.files:
        return jsonify({"error": "Missing multipart field 'file'"}), 400

    uploaded = request.files["file"]
    if not uploaded or not uploaded.filename:
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(uploaded.filename)
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are supported"}), 400

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_RAW_DIR / filename
    uploaded.save(dest)

    try:
        result = process_document(dest)
    except (FileNotFoundError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Ingest failed: {exc}"}), 500

    return jsonify({"ok": True, "filename": result["filename"]})


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "JSON body must include non-empty 'message'"}), 400

    try:
        answer = process_prompt(message)
    except DocumentNotLoadedError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Chat failed: {exc}"}), 500

    return jsonify({"answer": answer})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") in ("1", "true", "True")
    app.run(host="0.0.0.0", port=port, debug=debug)
