# Session C findings (laptop1: Ryzen 7 6800H, RTX 3060 Laptop 6 GB, 15 GB RAM)

Test set: 10,007 chunks (1,012 verses, all four Vedas, hi 5,428 / en 3,044 / sa 1,535), 360 generated questions
(hi, en, hi_roman, sa_quote x 90; each group 36 hi / 32 en / 22 sa gold chunks; 222 cross-language, 138 same-language).
Numbers are R@10 for the 'all' group, chunk / verse level, from results/C/results.csv. Latency is per question on laptop1.

1. C1a BM25, own tokenizer: .158 / .375, p50 0.6 ms. Useless across languages (chunk .03 cross vs .37 same) and for Hindi typed in Roman letters (.06).
2. C2a bge-m3 dense: .300 / .536, p50 42 ms (query encoding on GPU dominates). Sparse .200 / .472. Dense+sparse (RRF) .236 / .539: sparse adds nothing; drop it.
3. C3 fusion on questions as typed hurts overall: BM25+dense .253 / .536. It helps same-language questions (.47 vs dense .42) and hurts cross-language ones (.12 vs .23).
4. C4 question forms decide the outcome. Haiku rewrites each question into Devanagari Hindi and English; all forms fused (RRF k=60).
   Best: BM25+dense over original + Devanagari + English: .467 / .744, MRR@10 .253, R@50 .617 / .875. Chosen for candidates_top50.
   BM25+dense+sparse over the same forms: .450 / .764 (best verse level), R@50 .625 / .894.
5. Hindi typed in Roman letters: dense .01 / .10 as typed; .44 / .74 with the rewrites. The Devanagari form gives most of the gain; the English form adds the cross-language part.
6. Same- vs cross-language (chunk R@10): best setup .62 same / .37 cross (verse .78 / .72); dense as typed .42 / .23; BM25 .37 / .03.
7. By gold-chunk language (chunk R@10, best setup): Hindi .58, English .57, Sanskrit .12. Sanskrit chunks are found at verse level (.62) but rarely as the exact chunk.
8. Chunk level is strict: many works comment on the same verse. With the best setup, 28% of questions get the right verse in the top 10 but another work's chunk.
9. Cost and latency of C4: $0.114 per 1,000 questions (Haiku 5.5, measured); p50 1.71 s, p95 2.64 s per question, of which ~1.6 s is the Haiku call; retrieval alone ~45 ms.
9b. Keyword arm in fusion (B's run files, 10k): with dense on questions as typed, B's language-routed OpenSearch BM25 beats C1a
   (.267 / .561 vs .253 / .536); with dense over 3 forms both give .436 chunk (verse .714 vs .689). Running the keyword arm over
   the 3 forms as well adds more (.467 / .744). Expected best: language-routed BM25 over all question forms + bge-m3 dense (not run).
10. Verse groups: 2,755 of 20,380 verses (13.5%) share a group (same unaccented text in several Vedas). In the 10k corpus only 2 verses share a group, so verse scores do not change there.
11. Full-size run (every hi/sa/en chunk, 203,686): see the end of this file.
12. Paid spend, session C: $0.18 question writing + $0.04 rewrites = $0.22 (results/C/spend.log).

## Full-size run (corpus_size "full (203686)": every Hindi, Sanskrit and English chunk, same 360 questions)
13. Recall roughly halves at 20x the corpus. Best: BM25+dense+sparse over original + Devanagari + English: chunk .300 / verse .522
    (R@50 .417); the 10k candidate setup BM25+dense over 3 forms: .281 / .503. Dense as typed .150 / .308; BM25 as typed .094 / .236.
14. At full size bge-m3 sparse becomes worth keeping (+.019 chunk, +.019 verse over BM25+dense), unlike on 10k. Rank order otherwise holds:
    question rewrites >> fusion > dense > sparse > BM25.
15. Full size, best setup: same-language .47 / cross-language .19 (chunk), .62 / .46 (verse); Sanskrit gold chunks .08 chunk level;
    Hindi in Roman letters .36 / .50 (dense as typed: .00 / .02).
16. Verse groups matter at full size: verse R@10 .522 with groups vs .517 counting the exact verse only (.503 vs .492 for BM25+dense).
17. Latency at full size (laptop1, CPU search over 203k vectors, sparse and BM25 in memory): dense as typed p50 66 ms; best setup p50 1.83 s /
    p95 2.75 s (Haiku rewrite ~1.6 s). Full embedding: 203,686 chunks in ~26 min GPU time (8 parts, lock released between parts).
