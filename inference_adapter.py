#!/usr/bin/env python3
"""
Run inference using the base model + a saved LoRA adapter from treasury scoring SFT.

Usage:
  python scripts/inference_adapter.py --adapter ./adapter_out

  python scripts/inference_adapter.py --adapter ./adapter_out --prompt "chunk: The client expects USD inflows next quarter..."
"""

import argparse
import json
import os
import re
from datetime import datetime

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# System prompt used during treasury scoring inference.
SCORING_SYSTEM = (
    "You are a treasury scoring engine. Return exactly one JSON object on one line. "
    "No markdown, no extra text. Use exactly these keys: priority_score, scores, reasons, reason. "
    "scores and reasons must contain exactly: relevance, actionability, specificity, "
    "volume_of_transaction, currency, cashflow_type_and_context, hedging_policy, frequency, entity_info. "
    "priority_score is an integer from 0 to 100. Each score is an integer from 0 to 5, "
    "where 0 = not present, unrelated, or no value; 1 = very weak signal; 2 = limited value; "
    "3 = moderate value; 4 = strong value; 5 = highly relevant, specific, and decision-useful. "
    "Score based on the intrinsic quality and value of the data for treasury analysis, not writing style. "
    "relevance = fit to treasury/business context; actionability = helps trigger or support a decision; "
    "specificity = concrete treasury detail rather than generic background. "
    "The six treasury signal scores reflect how much useful detail is present about: "
    "transaction volume, currency, cashflow type and payment/collection context, hedging policy, "
    "frequency, and entity information. Reasons must justify and match the numeric scores. "
    "The JSON object must be complete, include the final top-level reason field, and end with a closing brace."
    "Do not describe missing information positively, and do not give high scores when the reason says "
    "the information is absent or vague. No missing or extra keys."
    "CRITICAL RULE: If the 'chunk' contains no specific treasury data, "
    "every single numeric score MUST be 0. Do not invent details. "
    "A score of 0 is the correct and expected behavior for irrelevant text. "
    "If you see 'nothing useful' or similar phrases, your priority_score must be 0."
    "Example of a 0-score input: 'chunk: hello how are you'\n"
    "Example output: {\"priority_score\": 0, \"scores\": {\"relevance\": 0, \"actionability\": 0, \"specificity\": 0, \"volume_of_transaction\": 0, \"currency\": 0, \"cashflow_type_and_context\": 0, \"hedging_policy\": 0, \"frequency\": 0, \"entity_info\": 0}, \"reasons\": {\"relevance\": \"Information absent\", \"actionability\": \"Information absent\", \"specificity\": \"Information absent\", \"volume_of_transaction\": \"Information absent\", \"currency\": \"Information absent\", \"cashflow_type_and_context\": \"Information absent\", \"hedging_policy\": \"Information absent\", \"frequency\": \"Information absent\", \"entity_info\": \"Information absent\"}, \"reason\": \"No treasury data.\"}\n"
    "Now score the following:"
)

DEFAULT_USER_PROMPT = (
    "chunk: The client expects USD 25 million of collections next quarter and "
    "JPY-denominated payments to suppliers. Treasury is considering layered "
    "forwards to reduce earnings volatility and improve cash flow certainty."
)

EXPECTED_TOP_KEYS = {"priority_score", "scores", "reasons", "reason"}
EXPECTED_SCORE_KEYS = {
    "relevance",
    "actionability",
    "specificity",
    "volume_of_transaction",
    "currency",
    "cashflow_type_and_context",
    "hedging_policy",
    "frequency",
    "entity_info",
}


def extract_first_json_object(text: str):
    """
    Finds the first balanced { } pair to handle nested JSON objects.
    """
    start_idx = text.find('{')
    if start_idx == -1:
        raise json.JSONDecodeError("No '{' found", text, 0)
    
    brace_count = 0
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            
        if brace_count == 0:
            potential_json = text[start_idx : i + 1]
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError:
                continue # Keep looking if this wasn't valid
                
    raise json.JSONDecodeError("No balanced JSON object found", text, 0)


def validate_output_schema(parsed: dict):
    """
    Validate the generated JSON against the training-time schema.
    Returns a list of human-readable issues.
    """
    issues = []

    if not isinstance(parsed, dict):
        return ["Top-level output is not a JSON object."]

    top_keys = set(parsed.keys())
    if top_keys != EXPECTED_TOP_KEYS:
        issues.append(
            f"Top-level keys mismatch. Expected {sorted(EXPECTED_TOP_KEYS)}, got {sorted(top_keys)}."
        )

    priority_score = parsed.get("priority_score")
    if not isinstance(priority_score, int) or not (0 <= priority_score <= 100):
        issues.append(f"'priority_score' must be an integer in [0, 100], got {priority_score!r}.")

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        issues.append("'scores' must be a JSON object.")
    else:
        score_keys = set(scores.keys())
        if score_keys != EXPECTED_SCORE_KEYS:
            issues.append(
                f"'scores' keys mismatch. Expected {sorted(EXPECTED_SCORE_KEYS)}, got {sorted(score_keys)}."
            )
        for k, v in scores.items():
            if not isinstance(v, int) or not (0 <= v <= 5):
                issues.append(f"'scores.{k}' must be an integer in [0, 5], got {v!r}.")

    reasons = parsed.get("reasons")
    if not isinstance(reasons, dict):
        issues.append("'reasons' must be a JSON object.")
    else:
        reason_keys = set(reasons.keys())
        if reason_keys != EXPECTED_SCORE_KEYS:
            issues.append(
                f"'reasons' keys mismatch. Expected {sorted(EXPECTED_SCORE_KEYS)}, got {sorted(reason_keys)}."
            )
        for k, v in reasons.items():
            if not isinstance(v, str):
                issues.append(f"'reasons.{k}' must be a string, got {type(v).__name__}.")

    overall_reason = parsed.get("reason")
    if not isinstance(overall_reason, str):
        issues.append(f"'reason' must be a string, got {type(overall_reason).__name__}.")

    return issues


def compute_priority_score(scores: dict) -> int:
    """
    Deterministically compute a priority score from the generated sub-scores.
    """
    relevance = scores.get("relevance", 0)
    specificity = scores.get("specificity", 0)
    actionability = scores.get("actionability", 0)
    volume = scores.get("volume_of_transaction", 0)
    currency = scores.get("currency", 0)
    cashflow = scores.get("cashflow_type_and_context", 0)
    hedging = scores.get("hedging_policy", 0)
    frequency = scores.get("frequency", 0)
    entity = scores.get("entity_info", 0)

    weighted_sum = (
        1.5 * specificity
        + 1.5 * relevance
        + actionability
        + volume
        + currency
        + cashflow
        + hedging
        + frequency
        + entity
    )

    return round(weighted_sum * 2)
    
def check_for_contradictions(parsed: dict):
    """Flags high scores paired with negative keywords."""
    negative_keywords = ["absent", "not specified", "missing", "none", "insufficient"]
    for key, score in parsed.get("scores", {}).items():
        reason_text = parsed.get("reasons", {}).get(key, "").lower()
        if score >= 3 and any(word in reason_text for word in negative_keywords):
            print(f"WARNING: Potential contradiction in '{key}'. Score is {score} but reason says: '{reason_text}'")


def main():
    parser = argparse.ArgumentParser(
        description="Inference with base model + treasury scoring LoRA adapter."
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./adapter_out",
        help="Path to saved adapter directory.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "User message content. Should follow the training format:\n"
            "chunk: ..."
        ),
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Enable stochastic sampling (default: False).",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of generations to run for analysis.",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Omit the system prompt. Not recommended for this adapter.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full parsed JSON output.",
    )
    args = parser.parse_args()
    
    if not args.verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        from transformers.utils import logging
        logging.set_verbosity_error()

        import logging as py_logging
        py_logging.getLogger("transformers").setLevel(py_logging.ERROR)
        py_logging.getLogger("peft").setLevel(py_logging.ERROR)

        # Disable tqdm globally
        os.environ["DISABLE_TQDM"] = "1"

    token = args.token or os.environ.get("HF_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if args.verbose:
        print(f"Loading base model and tokenizer on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if not args.verbose:
        from transformers.utils import logging
        logging.disable_progress_bar()

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        token=token,
        low_cpu_mem_usage=True,
    )

    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.to(device)
    model.eval()

    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.generation_config.typical_p = None

    user_content = args.prompt if args.prompt is not None else DEFAULT_USER_PROMPT

    if args.no_system:
        messages = [{"role": "user", "content": user_content}]
    else:
        messages = [
            {"role": "system", "content": SCORING_SYSTEM},
            {"role": "user", "content": user_content},
        ]

    date_string = datetime.now().strftime("%d %b %Y")

    try:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            date_string=date_string,
        )
    except TypeError:
        model_inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    input_ids = model_inputs["input_ids"]

    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.05,
    }

    for run_idx in range(args.num_runs):
        if args.num_runs > 1:
            print(f"\n=== Run {run_idx + 1}/{args.num_runs} ===")

        with torch.no_grad():
            outputs = model.generate(
                **model_inputs,
                **generate_kwargs,
            )

        prompt_len = input_ids.shape[1]
        answer_ids = outputs[0][prompt_len:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        try:
            parsed = extract_first_json_object(answer)

            # Validate only the model-generated schema first.
            issues = validate_output_schema(parsed)
            
            # 2. RUN THE CONTRADICTION CHECK (Add this line!)
            check_for_contradictions(parsed)

            # Overwrite priority_score with the deterministic score computed from sub-scores.
            scores = parsed.get("scores", {})
            if isinstance(scores, dict):
                parsed["priority_score"] = compute_priority_score(scores)

            if args.verbose:
                print(json.dumps(parsed, indent=2, ensure_ascii=False))
            else:
                print(parsed.get("priority_score", 0))

            if issues:
                print("\nSchema issues detected in model output:")
                for issue in issues:
                    print(f"- {issue}")

        except json.JSONDecodeError as e:
            print("\n(Output is not valid JSON)")
            print(f"JSON error: {e}")
            print("Raw output repr:")
            print(repr(answer))


if __name__ == "__main__":
    main()
