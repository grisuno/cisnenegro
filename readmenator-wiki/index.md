# Second Brain

*Last synthesized: 2026-10-07 | 37 files | 1 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `apex28.py`, `apex29.py`, `apex30.py`. Architecturally it is 1 layers, dominant utility (37 files) across 1 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Communities are self-contained in the resolved import graph; no cross-boundary bridges were recorded.

Open work clusters around documentation (95% file coverage), 0 security findings, 0 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 37 |
| Symbols | 1059 |
| Resolved imports | 0 |
| Languages | py, sh |
| Communities | 1 |
| Doc coverage | 95% (35/37 files) |
| Security findings | 0 |
| Estimated read cost | ~12614 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_cisnenegro_hc32yqxs
```

## Concept Wiki

- [root (37 files, cohesion 1.00)](./community_0_root.md)

## God Nodes

| File | Score |
|------|-------|
| `apex28.py` | 4.7 |
| `apex29.py` | 4.7 |
| `apex30.py` | 4.7 |
| `apex31.py` | 4.7 |
| `apex33.py` | 4.6 |

## Strongest Connections

- No cross-community connections recorded.

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
