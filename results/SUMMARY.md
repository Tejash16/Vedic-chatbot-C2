# C2 benchmark report

All numbers are for the 'all' group of 360 generated questions (hi, en, hi_roman, sa_quote x 90; 222 cross-language) unless stated. Full table: results/all_results.csv.

## Results

| setup | session | corpus | chunk R@5 | chunk R@8 | chunk R@10 | chunk MRR@10 | verse R@10 | p50 ms | p95 ms | $ per 1k q |
|---|---|---|---|---|---|---|---|---|---|---|
| C1a BM25 (bm25s, own tokenizer) | C | 10k | 0.133 |  | 0.158 | 0.110 | 0.375 | 0.6 | 0.7 | 0.00 |
| C1b OpenSearch, unstemmed | B | 10k | 0.131 |  | 0.156 | 0.109 | 0.372 | 10 | 19 | 0.00 |
| C1b OpenSearch, language analyzer routed by script | B | 10k | 0.161 |  | 0.194 | 0.124 | 0.431 | 9 | 17 | 0.00 |
| C2a bge-m3 dense | C | 10k | 0.233 |  | 0.300 | 0.174 | 0.536 | 42 | 62 | 0.00 |
| C2a bge-m3 sparse | C | 10k | 0.178 |  | 0.200 | 0.134 | 0.472 | 41 | 62 | 0.00 |
| C2a bge-m3 dense+sparse | C | 10k | 0.194 |  | 0.236 | 0.136 | 0.539 | 42 | 63 | 0.00 |
| C2b Qwen3-Embedding-0.6B dense (512) | A | 10k | 0.197 |  | 0.264 | 0.134 | 0.556 | 52 | 72 | 0.00 |
| C2b Qwen3-Embedding-0.6B dense (1024) | A | 10k | 0.214 |  | 0.269 | 0.147 | 0.558 | 53 | 73 | 0.00 |
| C2b Qwen3 dense + C1a (512) | A | 10k | 0.169 |  | 0.253 | 0.138 | 0.542 | 53 | 73 | 0.00 |
| C3 BM25 + bge-m3 dense | C | 10k | 0.192 |  | 0.253 | 0.140 | 0.536 | 43 | 63 | 0.00 |
| C3 BM25 + bge-m3 dense + sparse | C | 10k | 0.186 |  | 0.244 | 0.140 | 0.533 | 43 | 63 | 0.00 |
| C4 bge-m3 dense, 3 question forms | C | 10k | 0.317 |  | 0.419 | 0.229 | 0.672 | 1708 | 2640 | 0.11 |
| C4 BM25 + dense, 3 forms (candidates) | C | 10k | 0.372 |  | 0.467 | 0.253 | 0.744 | 1710 | 2643 | 0.11 |
| C4 BM25 + dense + sparse, 3 forms | C | 10k | 0.347 |  | 0.450 | 0.246 | 0.764 | 1711 | 2644 | 0.11 |
| C5 no reranker (candidates order) | B | 10k | 0.372 | 0.444 | 0.467 | 0.253 | 0.744 | 0.0 | 0.0 | 0.00 |
| C5 bge-reranker-v2-m3 | B | 10k | 0.450 | 0.486 | 0.497 | 0.330 | 0.778 | 1374 | 1523 | 0.00 |
| C5 Qwen3-Reranker-0.6B (512) | B | 10k | 0.411 | 0.461 | 0.486 | 0.301 | 0.747 | 4545 | 6530 | 0.00 |
| C5 claude-haiku-5-5 picks 8 (low effort) | B | 10k | 0.531 | 0.592 | 0.594 | 0.385 | 0.831 | 1451 | 3283 | 2.12 |
| C5 claude-haiku-5-5 picks 8 (medium effort) | B | 10k | 0.556 | 0.592 | 0.597 | 0.422 | 0.842 | 2527 | 4577 | 2.22 |
| C6 bge-m3 dense+sparse, 300 tokens | A | 10k | 0.194 |  | 0.233 | 0.136 | 0.536 | 41 | 63 | 0.00 |
| C6 bge-m3 dense+sparse, 150 tokens | A | 10k | 0.208 |  | 0.256 | 0.146 | 0.567 | 44 | 65 | 0.00 |
| C6 bge-m3 dense+sparse, 600 tokens | A | 10k | 0.189 |  | 0.236 | 0.136 | 0.531 | 40 | 61 | 0.00 |
| C6 bge-m3 dense+sparse, 300 tokens, no header | A | 10k | 0.206 |  | 0.242 | 0.151 | 0.556 | 41 | 62 | 0.00 |
| D bge-m3 dense, passages only (sub-corpus) | D | 1.5k | 0.406 |  | 0.467 | 0.280 | 0.679 | 38 | 58 | 0.00 |
| D + all kept sets, hi+en | D | 1.5k | 0.449 |  | 0.503 | 0.323 | 0.764 | 39 | 59 | 0.00 |
| D + all sets passing filters 1-2, hi+en | D | 1.5k | 0.509 |  | 0.612 | 0.373 | 0.794 | 40 | 60 | 0.00 |
| Full: bge-m3 dense | C | full | 0.128 |  | 0.150 | 0.103 | 0.308 | 66 | 86 | 0.00 |
| Full: BM25 + dense, 3 forms | C | full | 0.222 |  | 0.281 | 0.166 | 0.503 | 1816 | 2744 | 0.11 |
| Full: BM25 + dense + sparse, 3 forms | C | full | 0.253 |  | 0.300 | 0.182 | 0.522 | 1827 | 2752 | 0.11 |

## How to read the numbers

- Chunk level: the one gold chunk must come back. Verse level: any chunk of the gold verse, or of a verse with the same text in
  another Veda, counts. With 8 to 25 works commenting on each verse, chunk level is much stricter: with the best first-stage
  setup, 28% of questions find the right verse in the top 10 but another work's chunk on it.
- 10k = the 10,007-chunk test corpus (1,012 verses, all four Vedas). Full = every Hindi, Sanskrit and English chunk
  (203,686). 1.5k = session D's sub-corpus (1,499 chunks, 165 of the questions), so D's rows compare only with each other.
- Latency is per question on the machine named in the row. C5 latency comes on top of the candidate generation (C4, p50 1.71 s).
- All questions are generated (claude-haiku-5-5); there are no real user questions yet.

## Recommendations (ranked, best first)

**Keyword arm**
1. OpenSearch BM25 with a Hindi analyzer for Hindi/Sanskrit chunks and an English analyzer for English chunks, each question routed
   by its script (B): .194 / .431 alone, and the better keyword partner in fusion (.267 / .561 vs .253 / .536 with dense, as typed).
2. bm25s with our own tokenizer (C1a): .158 / .375, 0.6 ms in process, no extra service. Same as OpenSearch without stemming.
3. Never query several language fields at once and compare raw BM25 scores (B: .147 / .319).
- Keyword search cannot cross languages (chunk .03 to .06 cross-language). Use it only fused with dense, and run it on the rewritten
  question forms too: that lifts BM25 + dense from .436 to .467 (chunk R@10).

**Embedding model**
1. BAAI/bge-m3 dense: .300 / .536 (C), better than Qwen on Hindi (.41 vs .28 chunk), English (.40 vs .36) and Sanskrit-quote questions
   (.38 vs .28); query encoding ~36-42 ms; embeds 10k chunks in 2-3 min on a 4-6 GB laptop GPU.
2. Qwen/Qwen3-Embedding-0.6B: .264 / .556 at 512 tokens, .269 / .558 at 1,024 (its tokenizer cuts 40% of passages at 512, but the
   cut explains little). Slightly higher verse level, and much better on Hindi typed in Roman letters (.167 vs .011), a weakness the
   C4 rewrite already fixes for bge-m3 (.44). 13-15 min to embed 10k chunks. Fusing it with BM25 does not help (.253 / .542).
3. bge-m3 sparse: not alone (.200 / .472); as a fusion partner it hurts at 10k but adds +.019 at full size. Keep it only at full scale.

**Question form**
1. Rewrite every question with claude-haiku-5-5 into Devanagari Hindi and English and fuse all three forms (C4): BM25 + dense
   .253 / .536 as typed -> .467 / .744. Hindi typed in Roman letters goes from .01 to .44 (chunk) and .10 to .74 (verse).
   Cost $0.11 per 1,000 questions; latency +1.6 s (p50).
2. Original + Devanagari form only: BM25 + dense .408 / .742 (most of the gain); original + English only: .333 / .594.
3. As typed: fails on Roman-letter Hindi (.01) and cross-language questions (.23 dense).

**Reranker** (on the 50 C4 candidates, keep 8; chunk R@8, ceiling .617)
1. claude-haiku-5-5, medium effort: .592, MRR@8 .422, verse R@8 .836; p50 2.5 s, $2.22 per 1,000 questions. Low effort: same
   R@8, MRR@8 .385, p50 1.45 s, $2.12. The only reranker that helps cross-language (.351 -> .505) and Roman-letter Hindi.
2. bge-reranker-v2-m3 (free, local): .486, MRR@8 .329, p50 1.4 s on a 4 GB laptop GPU. Makes Roman-letter Hindi worse.
3. Qwen3-Reranker-0.6B: .461 at the 512-token cut, which truncates 74% of Hindi passages for its tokenizer; better than bge on
   English (.539 vs .484). Not settled for Hindi/Sanskrit (1,024-token run did not fit in 4 GB).
4. No reranker: .444. Candidate recall now limits results more than the reranker: 38% of questions have no gold chunk in the 50.

**Chunk size and header**
1. About 150 tokens (A, bge-m3 dense+sparse, overlap rule): .256 / .567, best at both chunk and verse level.
2. About 300 tokens: .233 / .536 with header, .242 / .556 without.
3. About 600 tokens (max_length 1024): .236 / .531, no gain; most records are short (8,356 chunks from 7,763 records).
- Header: removing it costs nothing at 300 tokens and helps cross-language verse recall (.500 -> .545). Drop it, or keep it only for
  display. Not yet tested: 150 tokens without header, and chunk size combined with the C4 forms and the reranker.

**Sets to index** (D, on its 1.5k sub-corpus, bge-m3 dense)
1. Index all question-answer sets in both Hindi and English that pass filters 1 and 2 (exact answer, known names/terms), without
   the "own passage in top 10" filter: chunk R@10 .467 -> .612, verse .679 -> .794. The gain is cross-language (.373 -> .627).
2. Sets that pass all three filters: .503 / .764. Filter 3 removes exactly the questions that open a new route to their passage.
3. Source-language questions only: .461 / .721, no gain. Best-K per passage: best 5 gives .479; more equals all kept sets.
- Estimated cost for the whole library: ~1.07 M sets (1.51 M without filter 3), about $112 on the Haiku Batch API
  (../local/final/sets-estimate.md). Sets do not fix Roman-letter Hindi; the question rewrite does.

**Proposed pipeline from these results**: ~150-token chunks without header -> rewrite the question (Haiku, Devanagari + English)
-> language-routed BM25 + bge-m3 dense (+ sparse at full scale) over passages and indexed sets, RRF k = 60, top 50 ->
claude-haiku-5-5 picks 8 (low effort when latency matters). Expected per-question cost about $2.3 per 1,000 questions and 3 to 4 s latency, both dominated by the two Haiku calls.

## Full size

At 203,686 chunks recall roughly halves (best: BM25 + dense + sparse over 3 forms .300 / .522; dense as typed .150 / .308), and
bge-m3 sparse becomes worth fusing (+.019). The ranking of setups holds. Verse groups (the same mantra in several Vedas) raise
verse R@10 from .517 to .522. Same-language .47 vs cross-language .19 at chunk level: the reranker and the sets, both measured on
10k / 1.5k only, target exactly this gap.

## Failure cases (patterns)

Over 46 setups with per-question rankings (A, B, C on the 10k corpus), 104 of 360 questions have the gold chunk in nobody's top 10
and 7 have the right verse in nobody's top 20. Sanskrit gold chunks dominate: 51 of the 88 Sanskrit-chunk questions (58%) are missed
by every setup, against 15% for Hindi and 25% for English. In the 20 worst cases (listed with text in the local report only):
- Generic theme questions ("the duty of a king like Agni", "how Indra treats his people") retrieve a sound answer on another verse:
  the question does not pin down the verse. Partly a limit of generated questions; real questions will show how common this is.
- Sanskrit commentary (9 of 20), often grammar or etymology notes, asked about in Hindi, English or Roman letters.
- Hindi typed in Roman letters (11 of 20), even after the rewrite, usually together with a Sanskrit or English gold chunk.
- Exact-term questions about textual notes (a spelling variant, a causative verb form) written in transliteration, while the
  passage has the term in Devanagari.

## What to test next

1. Real user questions (group `real`): every number here comes from generated questions.
2. The proposed pipeline end to end at full size: language-routed BM25 over the rewritten forms + dense + sparse + sets, then the
   Haiku reranker on the full-size top 50. Only the pieces were measured.
3. Sanskrit passages: chunk R@10 .12 (10k) and .08 (full) against .58 for Hindi. Try indexing each Sanskrit chunk with its Hindi
   counterpart from the same author, or returning all chunks of a retrieved verse and reranking them.
4. Candidate recall: 38% of gold chunks are not in the top 50. Try top 100 for the reranker, verse-level expansion (all chunks of the
   top verses), and sets on the full corpus.
5. Qwen3 models with a 1,024-token cut on a larger GPU (both embedder and reranker were truncated for Devanagari at 512).
6. One Haiku call for rewrite + rerank, or a cheaper transliteration step for Roman-letter Hindi, to cut the ~3-4 s latency.
7. Scholar review of the 120 blinded sets (../local/final/sets-for-review.csv) before generating sets for the whole library.
8. The open questions for the society (../local/final/open-questions.md): two possible author pairs, Yajurveda 25/47-48, Samaveda
   spellings in verse groups.

## Machines

- laptop1 | AMD Ryzen 7 6800H with Radeon Graphics | NVIDIA GeForce RTX 3060 Laptop GPU 6GB | 15GB RAM: sessions C, D
- laptop2 | 11th Gen Intel(R) Core(TM) i5-11400H @ 2.70GHz | NVIDIA GeForce RTX 3050 Laptop GPU 4GB | 8GB RAM: sessions A, B
