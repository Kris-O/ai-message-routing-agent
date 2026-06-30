# Round 21 — eval on Qwen3-4B-Instruct-2507 (tool-calling agent)

Same stress set as round-12 (74 deliberately messy messages × 3 = **222 live-endpoint calls**), now with
the agent sending mail via a real `send_email` tool call (tool/function calling), model
**Qwen3-4B-Instruct-2507** (Unsloth GGUF `UD-Q4_K_XL`), CPU.

| Metric | Result |
|---|---|
| run-level accuracy | **216/222 = 97.3%** |
| case-level (majority of 3) | 72/74 = 97.3% |
| determinism (3/3 identical) | **74/74 = 100%** |
| upstream errors / failed tool calls | **0/222** |
| prompt-injection leaks | **0** (18/18 attempts deflected) |
| per department | KADRY / HR / HELPDESK / IT = 100%, INNE = 83% |

Two case-level misses, both in the INNE fallback bucket (the inherently fuzzy "everything else" class):
- `n05` expected INNE → HELPDESK (vague help request)
- `n12` expected INNE → HR (off-topic)

**Takeaways.** Function calling is reliable on this model — **0 failed tool calls across 222**, so no
fallback path is needed. Determinism (100%) and zero injection leaks are preserved, and accuracy is
**higher than the previous deterministic variant** (93.2% on qwen2.5:3b, same test set). The earlier
qwen2.5:3b build could not emit tool calls at all; switching to a non-thinking instruct model that does
both tool-calling and Polish well is what closed the gap.
