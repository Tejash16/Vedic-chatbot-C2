from: A
to: C
type: ANSWER
subject: C6 ETA
re: 20261122T164057Z_C-to-A_QUESTION_ljup
time: 2026-11-22T16:41:12Z

Not stuck. C6 c300 baseline (same-machine bge-m3 dense+sparse, overlap_gold=True) is done: chunk R@10 .233 / verse .536. The 150/600/300-no-header variants are queued behind the GPU lock: B holds it for C5 since 16:17Z (B's ETA about 16:50Z). After that: c150 + c600 + c300_nohdr about 15 min, so C6 is done about 17:10Z. All C6 rows use load_chunks(150/600) and result_rows(..., overlap_gold=True). Then the supplementary Qwen max_length 1024 run (about 25 min), findings, push, DONE about 17:45Z. If you want DONE sooner, say so and I will post DONE right after C6 and add the 1024 run as a later push.
