# C2 benchmark kit

Four Claude Code sessions on two laptops run one benchmark and report to one place.

| Session | Laptop | Work |
|---|---|---|
| C (lead) | Tejash | Data audit, test set, shared code, bge-m3 retrieval, final results |
| D | Tejash | Question-and-answer sets |
| A | Kedar | Chunks on laptop 2, Qwen embedding model, chunk-size variants |
| B | Kedar | OpenSearch keyword arm, rerankers |

## Files

- `C2_SPEC.md`: what the benchmark measures and who does which step
- `PROTOCOL.md`: how the sessions coordinate
- `roles/A.md` to `roles/D.md`: each session's own task list
- `bus/bus.py`: the message bus (messages travel through this Git repo)

## Folder layout on each laptop

```
c2/
  local/            never in Git
    data/           the Excel files
    work/           derived files
    final/          reports
    hf-cache/       downloaded models
    venv/           Python environment
    .env            VEDIC_ANTHROPIC_API_KEY=...
  session-C/        a clone of this repo (A and B on Kedar's laptop)
  session-D/        a second clone
```

## What must never be pushed

The Excel files, anything derived from their text, and the API key. The repo must be private.

## Starting a session

Open the session's folder in its own VS Code window, start Claude Code, and paste:

```
You are session C. Read PROTOCOL.md, C2_SPEC.md and roles/C.md in this folder, then start. Follow the protocol exactly.
```

Change the letter for the other sessions.

## If a session stops

Sessions do not wake up on their own when a message arrives. If one has stopped, type: `Check the bus and continue.`
