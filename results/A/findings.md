# Session A findings: C2b (Qwen3-Embedding-0.6B) and C6 (chunk variants), laptop 2

Setup: laptop 2 (i5-11400H, RTX 3050 Laptop 4 GB, 8 GB RAM), 10k corpus (10,007 chunks), 360 generated questions, no paid calls. Figures are R@10 over all questions, chunk / verse level; C6 uses `overlap_gold=True`.

1. **Chunks on laptop 2 match laptop 1.** `python -m src.common.chunk --rebuild` printed MANIFEST MATCH (10,007 chunks, same SHA-256).
2. **C2b, Qwen3 dense (official, max_length 512):** .264 / .556 (MRR@10 .134 / .341). Fused with C1a BM25 (RRF k=60): .253 / .542. C1a alone on this laptop: .158 / .375.
3. **Qwen's tokenizer needs about 2.4 times the bge-m3 tokens on Devanagari** (median per chunk: hi 580 vs 238, sa 452 vs 208, en 91 vs 94). At 512 it cuts off 40% of passages (hi 61%, sa 43%); bge-m3 cuts off none.
4. **Without the cut-off, Qwen barely improves.** Supplementary run at max_length 1024 (0.4% cut off): .269 / .558 (MRR .147 / .348); hi chunk .233 → .278. The cut-off is not why Qwen trails.
5. **bge-m3 dense beats Qwen dense on the same 300-token chunks:** .300 / .539. Chunk R@10 by group: en .400 vs .356, hi .411 vs .278, sa_quote .378 vs .278. **The one exception is Hindi typed in Roman letters:** Qwen .167 / .333 against bge-m3 .011 / .100. hi_roman is near zero for every bge-m3 setup.
6. **BM25 fusion helps same-language questions and hurts cross-language ones** (Qwen chunk R@10: same .428 → .478, cross .162 → .113; 222 of the 360 questions are cross-language). C1a alone scores .027 on cross-language questions.
7. **bge-m3 sparse pulls the fusion down at chunk level, at every chunk size.** 300 with header: dense .300, sparse .200, dense+sparse .233. At verse level the gap is smaller (.539 / .472 / .536).
8. **C6, bge-m3 dense+sparse:** 300 with header (baseline) .233 / .536; **150: .256 / .567**; 600 (max_length 1024) .236 / .531; 300 without header .242 / .556.
9. **150-token chunks are best.** The gain holds at verse level, where the overlap rule plays no part. 600 adds nothing, since most records are short (8,356 chunks from 7,763 records). Removing the header costs nothing and helps cross-language questions (verse R@10 .500 → .545).
10. **Latency per question on this laptop** (one quiet window, all A setups back-to-back):
    - Qwen setups: p50 52–53 ms, p95 72–74 ms.
    - bge-m3 dense+sparse: p50 40–44 ms, p95 61–65 ms (query encoding p50 36 ms).
    - BM25: 0.5 ms.

    A first pass taken while the other session loaded the shared 8 GB laptop measured 215–275 ms, so compare latency only within one quiet window.
11. **Passage encoding on the RTX 3050:** bge-m3 2.3–3.2 min per variant (10k chunks). Qwen 13.3 min at 512 and 15.0 min at 1024, because its Devanagari token sequences are longer.
12. **Recommendation:**
    - Keep bge-m3 as the embedding model. It is more accurate on hi, en and sa, and faster.
    - Add Qwen only if the Devanagari rewrite of C4 does not fix hi_roman, since hi_roman is Qwen's one strength.
    - Use chunks of about 150 tokens.
    - The header is optional; dropping it is safe.
    - Next to test: 150 tokens without the header, and bge-m3 dense alone against dense+sparse at full size.
13. **Files:** top-50 chunk IDs per question are in `results/A/runs/<run_id>.jsonl`; C6 IDs refer to chunks_150 / chunks_600 / chunks.parquet. The code is in `src/A/`.
