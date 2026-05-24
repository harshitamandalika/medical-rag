from __future__ import annotations
import json
from pathlib import Path

PROMPTS_DIR          = Path("prompts")
SYSTEM_PROMPT_FILE   = PROMPTS_DIR / "system_prompt_v1.txt"
FEW_SHOT_FILE        = PROMPTS_DIR / "few_shot_examples.json"

_system_prompt_cache: str | None = None
_few_shot_cache:      str | None = None


def _load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache is None:
        _system_prompt_cache = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    return _system_prompt_cache


def _load_few_shot_block() -> str:
    global _few_shot_cache
    if _few_shot_cache is None:
        examples = json.loads(FEW_SHOT_FILE.read_text(encoding="utf-8"))
        lines = ["EXAMPLES OF CORRECT OUTPUT FORMAT:\n"]
        for i, ex in enumerate(examples, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Question: {ex['question']}")
            lines.append(f"Context summary: {ex['context_summary']}")
            lines.append(f"Output: {json.dumps(ex['answer'], indent=2)}")
            lines.append("")
        _few_shot_cache = "\n".join(lines)
    return _few_shot_cache


def build_context_block(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant abstracts were retrieved for this query."

    lines = ["RETRIEVED ABSTRACTS:\n"]
    for chunk in chunks:
        meta  = chunk["metadata"]
        pmid  = meta.get("pmid", "unknown")
        title = meta.get("title", "")
        year  = meta.get("year", "")
        journal = meta.get("journal", "")

        lines.append(f"[{chunk['rank']}] PMID: {pmid}")
        if title:
            lines.append(f"    Title: {title}")
        if journal and year:
            lines.append(f"    Source: {journal}, {year}")
        lines.append(f"    Text: {chunk['text']}")
        lines.append("")

    return "\n".join(lines)


def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    system_prompt = _load_system_prompt() + "\n\n" + _load_few_shot_block()
    context_block = build_context_block(chunks)

    user_message = (
        f"{context_block}\n"
        f"QUESTION:\n{question}\n\n"
        f"Respond with a JSON object only. No text outside the JSON."
    )

    return system_prompt, user_message