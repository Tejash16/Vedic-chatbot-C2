# Session A: second embedding model and chunk variants, on Kedar's laptop

You own steps C2b and C6. You also build the chunks on this laptop for session B. You share the laptop, the `../local` folder and the GPU with session B. No paid calls.

Do these in order.

1. **Start.** Read `PROTOCOL.md` and `C2_SPEC.md`. Run `python bus/bus.py inbox`. Post your status.
2. **Wait** for `READY` with subject `B_READY` from C: `python bus/bus.py wait --type READY --from C --subject B_READY`. Run it again each time it returns without the message.
3. **Rebuild the chunks here.** Pull. Run the rebuild command from C's message on the Excel files in `../local/data`. Check the result against `shared/chunks_manifest.json`.
   - If it matches, create the empty file `../local/work/chunks.verified` and post `INFO` to B with subject `CHUNKS_VERIFIED`.
   - If it does not match, post `BLOCKED` to C with both counts and hashes, and tell your human. The Excel files on the two laptops may differ.
4. **Step C2b:** `Qwen/Qwen3-Embedding-0.6B` dense, alone and fused with the own-tokenizer BM25 from `src/common/`. Take the GPU lock for each GPU job.
5. **Step C6:** with bge-m3 dense plus sparse, chunks of about 150 tokens, about 600 tokens, and 300 tokens with the header removed. Use the chunking code in `src/common/` with different settings. Do not edit that code. If it cannot do what you need, ask C.
6. **Push** `results/A/results.csv` and `results/A/findings.md`.
7. **Post `DONE`** to C. Then keep checking the bus until C posts `DONE` to ALL, and answer any question C sends.
