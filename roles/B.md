# Session B: OpenSearch arm and rerankers, on Kedar's laptop

You own steps C1b and C5. You share the laptop, the `../local` folder and the GPU with session A. Spending limit: 5 US dollars, for the Claude reranker only.

Do these in order.

1. **Start.** Read `PROTOCOL.md` and `C2_SPEC.md`. Run `python bus/bus.py inbox`. Post your status.
2. **Wait** for `READY` with subject `B_READY` from C. Then wait for session A to verify the chunks on this laptop: `python bus/bus.py wait --type INFO --from A --subject CHUNKS_VERIFIED`. Do not build the chunks yourself. Run each wait again whenever it returns without the message.
3. **Step C1b.** If Docker is available, start OpenSearch in a container and run BM25 with the Hindi analyzer as the spec says. If Docker is not available, post `INFO` to C with subject `C1B_SKIPPED` and move on.
4. **Wait** for `READY` with subject `CANDIDATES_READY` from C, then pull.
5. **Step C5:** the three rerankers on `shared/candidates_top50.jsonl`, keeping 8. Take the GPU lock for the two local rerankers and release it straight after. Show your human the cost estimate before the Claude reranker.
6. **Push** `results/B/results.csv` and `results/B/findings.md`.
7. **Post `DONE`** to C. Then keep checking the bus until C posts `DONE` to ALL, and answer any question C sends.
