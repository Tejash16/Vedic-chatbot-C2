# Session D: question-and-answer sets, on Tejash's laptop

You own Stage D. You share the laptop, the `../local` folder and the GPU with session C. Spending limit: 10 US dollars.

Do these in order.

1. **Start.** Read `PROTOCOL.md` and `C2_SPEC.md`. Run `python bus/bus.py inbox`. Post your status.
2. **Wait** for `READY` with subject `B_READY` from C: `python bus/bus.py wait --type READY --from C --subject B_READY`. Run it again each time it returns without the message. While you wait, write and dry-run your Stage D code on made-up passages, with no paid calls.
3. **Pull.** Session C built `../local/work/chunks.parquet` on this laptop. Check it against `shared/chunks_manifest.json` using the code in `src/common/`.
4. **Stage D, steps 1 to 6** from the spec. Show your human the cost estimate before each paid step. Take the GPU lock for each GPU job and release it straight after, because C needs the GPU too.
5. **Write** `../local/final/sets-estimate.md`, `../local/final/sets-for-review.csv` and its answer key.
6. **Push** `results/D/results.csv` and `results/D/findings.md`. The findings must give: sets per 100 words, the share removed by each filter, the estimated total sets and cost, and whether indexing sets improved recall. No source text.
7. **Post `DONE`** to C. Then keep checking the bus until C posts `DONE` to ALL, and answer any question C sends.
