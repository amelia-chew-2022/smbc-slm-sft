#!/usr/bin/env python3
"""Small local HTTP wrapper around inference_adapter-style scoring."""

import argparse
import json
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_adapter import (
    DEFAULT_USER_PROMPT,
    MODEL_ID,
    SCORING_SYSTEM,
    compute_priority_score,
    extract_first_json_object,
    validate_output_schema,
)


MODEL = None
TOKENIZER = None
DEVICE = None
LOCK = threading.Lock()
ADAPTER_PATH = None
MAX_NEW_TOKENS = 1024


def load_runtime(adapter_path: str, token: str | None = None, max_new_tokens: int = 1024) -> None:
    global MODEL, TOKENIZER, DEVICE, ADAPTER_PATH, MAX_NEW_TOKENS

    ADAPTER_PATH = str(Path(adapter_path).resolve())
    MAX_NEW_TOKENS = max_new_tokens
    token = token or os.environ.get("HF_TOKEN")

    if torch.cuda.is_available():
        DEVICE = "cuda"
        dtype = torch.bfloat16
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        DEVICE = "mps"
        dtype = torch.float16
    else:
        DEVICE = "cpu"
        dtype = torch.float32

    print(f"Loading scorer model on {DEVICE} from {ADAPTER_PATH}...")

    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )

    MODEL = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    MODEL.to(DEVICE)
    MODEL.eval()

    if hasattr(MODEL, "generation_config") and MODEL.generation_config is not None:
        MODEL.generation_config.temperature = None
        MODEL.generation_config.top_p = None
        MODEL.generation_config.top_k = None
        MODEL.generation_config.typical_p = None

    print("Scorer model ready.")


def _build_messages(chunk_text: str) -> list[dict[str, str]]:
    user_content = chunk_text.strip() or DEFAULT_USER_PROMPT
    if not user_content.lower().startswith("chunk:"):
        user_content = f"chunk: {user_content}"
    return [
        {"role": "system", "content": SCORING_SYSTEM},
        {"role": "user", "content": user_content},
    ]


def _generate_raw_output(chunk_text: str) -> str:
    messages = _build_messages(chunk_text)
    date_string = datetime.now().strftime("%d %b %Y")

    try:
        model_inputs = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            date_string=date_string,
        )
    except TypeError:
        model_inputs = TOKENIZER.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    model_inputs = {k: v.to(DEVICE) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]

    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "pad_token_id": TOKENIZER.pad_token_id,
        "eos_token_id": TOKENIZER.eos_token_id,
        "repetition_penalty": 1.05,
    }

    with LOCK, torch.no_grad():
        outputs = MODEL.generate(**model_inputs, **generate_kwargs)

    prompt_len = input_ids.shape[1]
    answer_ids = outputs[0][prompt_len:]
    return TOKENIZER.decode(answer_ids, skip_special_tokens=True).strip()


def score_chunk(chunk_text: str) -> dict:
    try:
        raw_output = _generate_raw_output(chunk_text)
        parsed = extract_first_json_object(raw_output)
        issues = validate_output_schema(parsed)

        scores = parsed.get("scores", {})
        if isinstance(scores, dict):
            parsed["priority_score"] = compute_priority_score(scores)

        parsed["schema_issues"] = issues
        parsed["ok"] = not issues
        return parsed
    except Exception as exc:
        return {
            "priority_score": 0,
            "scores": {},
            "reasons": {},
            "reason": f"Scoring failed: {exc}",
            "schema_issues": [str(exc)],
            "ok": False,
        }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[scorer] {self.address_string()} - {format % args}")

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(
                {
                    "ok": True,
                    "model_id": MODEL_ID,
                    "adapter_path": ADAPTER_PATH,
                }
            )
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except json.JSONDecodeError as exc:
            self._send_json({"error": f"Invalid JSON body: {exc}"}, status=400)
            return

        path = self.path.rstrip("/")
        if path == "/score":
            chunk = body.get("chunk")
            if not isinstance(chunk, str):
                self._send_json({"error": "Expected string field 'chunk'."}, status=400)
                return
            self._send_json(score_chunk(chunk))
            return

        if path == "/score-batch":
            items = body.get("chunks")
            if not isinstance(items, list):
                self._send_json({"error": "Expected list field 'chunks'."}, status=400)
                return

            total = len(items)
            batch_start = time.time()
            print(f"[scorer] Starting batch scoring for {total} chunk(s)", flush=True)

            results = []
            for idx, item in enumerate(items, start=1):
                if isinstance(item, str):
                    chunk_text = item
                elif isinstance(item, dict):
                    chunk_text = str(item.get("chunk", ""))
                else:
                    chunk_text = ""

                result = score_chunk(chunk_text)
                results.append(result)

                if idx == 1 or idx == total or idx % 5 == 0:
                    elapsed = time.time() - batch_start
                    print(
                        f"[scorer] Scored chunk {idx}/{total} "
                        f"priority_score={result.get('priority_score', 0)} "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )

            total_elapsed = time.time() - batch_start
            print(
                f"[scorer] Completed batch scoring for {total} chunk(s) "
                f"in {total_elapsed:.1f}s",
                flush=True,
            )
            self._send_json({"results": results})
            return

        self._send_json({"error": "Not found"}, status=404)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local treasury scorer server.")
    parser.add_argument(
        "--adapter",
        type=str,
        default=str(Path(__file__).resolve().parent / "adapter_out"),
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--token", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()

    load_runtime(
        adapter_path=args.adapter,
        token=args.token,
        max_new_tokens=args.max_new_tokens,
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Scorer service listening on http://{args.host}:{args.port}")
    print("Endpoints: GET /health, POST /score, POST /score-batch")
    server.serve_forever()


if __name__ == "__main__":
    main()
