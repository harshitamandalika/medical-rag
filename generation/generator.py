from __future__ import annotations
import json
import os
import re
import time
from dataclasses import dataclass

import google.generativeai as genai
from google.api_core.exceptions import TooManyRequests, ServiceUnavailable
from generation.prompt_builder import build_prompt


MODEL_NAME  = "models/gemini-2.5-flash"
MAX_TOKENS  = 4096
TEMPERATURE = 0.1


@dataclass
class GenerationResult:
    answer:            str
    cited_pmids:       list[str]
    confidence:        str
    confidence_reason: str
    latency_ms:        float
    prompt_tokens:     int
    completion_tokens: int
    model:             str
    raw_response:      str


def _configure_client() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set"
        )
    genai.configure(api_key=api_key, transport="rest")


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text  = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    return text.strip()


def _parse_response(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Model returned an empty response")

    cleaned = _strip_fences(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        try:
            recoverable = re.sub(r',\s*"[^"]*$', "", cleaned)
            recoverable = re.sub(r':\s*"[^"]*$', ': ""', recoverable)
            if not recoverable.rstrip().endswith("}"):
                recoverable = recoverable.rstrip().rstrip(",") + "}"
            return json.loads(recoverable)
        except (json.JSONDecodeError, Exception):
            raise json.JSONDecodeError(str(e), cleaned, e.pos)


def _call_with_retry(model, contents: list, max_attempts: int = 4) -> object:
    delay = 60
    for attempt in range(1, max_attempts + 1):
        try:
            return model.generate_content(contents=contents)
        except TooManyRequests as e:
            error_str = str(e)
            if "PerDay" in error_str or "per_day" in error_str.lower():
                print("  generator: daily quota exhausted, retrying will not help.")
                raise
            if attempt == max_attempts:
                raise
            print(
                f"  generator: 429 rate limit hit (attempt {attempt}/{max_attempts}), "
                f"waiting {delay}s before retry"
            )
            time.sleep(delay)
            delay *= 2
        except ServiceUnavailable as e:
            if attempt == max_attempts:
                raise
            print(
                f"  generator: 503 service unavailable (attempt {attempt}/{max_attempts}), "
                f"waiting {delay}s before retry"
            )
            time.sleep(delay)
            delay *= 2


class Generator:
    def __init__(self):
        _configure_client()
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            generation_config=genai.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )

    def generate(self, question: str, chunks: list[dict]) -> GenerationResult:
        system_prompt, user_message = build_prompt(question, chunks)
        full_prompt = system_prompt + "\n\n" + user_message

        t0       = time.time()
        response = _call_with_retry(
            self.model,
            contents=[{"role": "user", "parts": [full_prompt]}],
        )
        latency_ms = (time.time() - t0) * 1000
        raw_text   = response.text

        try:
            parsed = _parse_response(raw_text)
        except (json.JSONDecodeError, ValueError):
            print("  generator: JSON parse failed on first attempt, retrying with correction prompt")
            correction = (
                "Your previous response was not valid JSON. "
                "Respond with a JSON object only. No explanation, no markdown, no preamble.\n\n"
                + user_message
            )
            retry_response = _call_with_retry(
                self.model,
                contents=[{"role": "user", "parts": [system_prompt + "\n\n" + correction]}],
            )
            latency_ms += (time.time() - t0) * 1000
            raw_text    = retry_response.text
            parsed      = _parse_response(raw_text)

        usage             = getattr(response, "usage_metadata", None)
        prompt_tokens     = getattr(usage, "prompt_token_count",     0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0

        print(
            f"  generator: done in {latency_ms:.0f}ms "
            f"({prompt_tokens} prompt tokens, {completion_tokens} completion tokens)"
        )

        return GenerationResult(
            answer            = parsed.get("answer", ""),
            cited_pmids       = parsed.get("cited_pmids", []),
            confidence        = parsed.get("confidence", "low"),
            confidence_reason = parsed.get("confidence_reason", ""),
            latency_ms        = latency_ms,
            prompt_tokens     = prompt_tokens,
            completion_tokens = completion_tokens,
            model             = MODEL_NAME,
            raw_response      = raw_text,
        )