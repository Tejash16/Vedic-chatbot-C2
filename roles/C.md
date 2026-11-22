# Session C: lead, on Tejash's laptop

You own the data audit, the shared test set and code, the main retrieval line, and the final results. Spending limit: 5 US dollars.

Do these in order.

1. **Start.** Read `PROTOCOL.md` and `C2_SPEC.md`. Run `python bus/bus.py inbox`. Post your status.
2. **Stage A.** Run the data audit. Write `../local/final/data-audit.md`. Post an `INFO` to ALL with table names, row counts and how the tables link, with no source text. Then stop, show your human the summary and your column assumptions, and wait for a go.
3. **Stage B.** Build the chunks and the test set. Write `src/common/` so that any session can rebuild the same chunks with one command and score a run with one function. Write `shared/chunks_manifest.json`, `shared/test_questions.jsonl`, `shared/config.yaml` and `requirements.lock`. Push them. Post `READY` with subject `B_READY` to ALL, and put the rebuild command in the message.
4. **Stage C, your steps:** C1a, C2a, C3 and C4 from the spec. Take the GPU lock for each GPU job.
5. **Candidates.** From your best fused setup, write `shared/candidates_top50.jsonl` and push it. Post `READY` with subject `CANDIDATES_READY` to B.
6. **Push your results:** `results/C/results.csv` and `results/C/findings.md`.
7. **Stay available.** While A, B and D work, keep checking the bus and answer their questions quickly. You are the only one who changes `src/common/` and `shared/`.
8. **Final results.** When A, B and D have each posted `DONE`, follow "Final results" in the spec. Then post `DONE` to ALL and tell your human which files in `../local/final` to send on.

If a session has not posted `DONE` and its status has not moved for 30 minutes, post a `QUESTION` to it and tell your human, who can nudge that session.
