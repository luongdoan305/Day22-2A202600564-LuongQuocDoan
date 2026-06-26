# Day 22 Evidence Notes

Generated artifacts in this folder:

- `02_ab_routing_log.txt`: console output for Prompt Hub A/B routing, with V1/V2 labels.
- `03_ragas_scores.txt`: score table output generated in offline fallback mode because networked RAGAS evaluator calls were blocked in the sandbox.
- `03_ragas_report.json`: JSON score report for both prompt versions.
- `04_pii_demo_log.txt`: PII redaction demo output.
- `04_json_demo_log.txt`: JSON repair demo output.

Prompt Hub URLs from the successful run:

- V1: https://smith.langchain.com/prompts/luong-quoc-doan-day22-rag-prompt-v1/43bd3b5e
- V2: https://smith.langchain.com/prompts/luong-quoc-doan-day22-rag-prompt-v2/7c9e198d

Manual screenshots still required for full rubric evidence:

- `01_langsmith_traces.png`: LangSmith Runs page showing at least 50 traces.
- `02_prompt_hub.png`: LangSmith Prompt Hub page showing both prompt versions.
- `03_ragas_scores.png`: terminal screenshot showing the RAGAS/offline score table.
