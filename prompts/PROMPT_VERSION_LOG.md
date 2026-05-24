# Prompt Version Log

## v1 — initial version

File: system_prompt_v1.txt
Date: May 2026

Changes: baseline prompt with citation rules, JSON output schema,
confidence field, and two few-shot examples.

RAGAS scores:
  faithfulness       : 0.9027
  answer_relevancy   : 0.5976
  context_precision  : 0.2133
  context_recall     : 0.0867

Notes: baseline on 574-chunk corpus across 6 conditions.
Low answer_relevancy driven by queries where corpus lacked
direct evidence. Low context_recall reflects corpus size
rather than retrieval quality.