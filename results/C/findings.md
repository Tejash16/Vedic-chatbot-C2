# Session C findings (laptop1: Ryzen 7 6800H, RTX 3060 Laptop 6 GB, 15 GB RAM)

Values are R@10, chunk / verse level, 'all' group of 360 questions (222 cross-language), from results/C/results.csv.
1. C1a BM25 (own tokenizer): 10k .158 / .375, p50 0.6 ms. Fails across languages (chunk .03 cross vs .37 same) and on Roman-letter Hindi (.06).
2. C2a bge-m3: dense .300 / .536 (p50 42 ms, GPU query encoding); sparse .200 / .472; dense+sparse RRF .236 / .539.
3. C3 fusion on questions as typed: BM25+dense .253 / .536. Helps same-language (.47 vs .42), hurts cross-language (.12 vs .23).
4. C4 question forms (Haiku rewrite to Devanagari Hindi and to English, all forms fused) is the largest gain: BM25+dense over 3 forms
   .467 / .744 (MRR .253, R@50 .617): the setup used for candidates_top50. Roman-letter Hindi: .01 -> .44 (verse .10 -> .74).
5. C4 cost $0.114 per 1,000 questions; latency p50 1.71 s / p95 2.64 s, ~1.6 s of it the Haiku call (retrieval ~45 ms).
6. Same vs cross language, best 10k setup: chunk .62 / .37, verse .78 / .72. Sanskrit gold chunks are hard: chunk .12 (Hindi .58, English .57), verse .62.
7. Chunk level is strict: 28% of questions get the right verse in the top 10 but another work's chunk on it.
8. Keyword arm (B's runs): B's language-routed OpenSearch beats C1a in fusion with dense (.267 / .561 vs .253 / .536); running the keyword
   arm on all 3 question forms adds more (.467 vs .436). Untested best: language-routed BM25 over all forms + bge-m3 dense (+ sparse).
9. Full size (all 203,686 hi/sa/en chunks, same questions): recall roughly halves. Best: BM25+dense+sparse over 3 forms .300 / .522
   (R@50 .417); BM25+dense over 3 forms .281 / .503; dense as typed .150 / .308; BM25 .094 / .236. At full size sparse helps (+.019).
10. Full size, best setup: same / cross language chunk .47 / .19, verse .62 / .46; Roman-letter Hindi .36 / .50; p50 1.83 s, p95 2.75 s.
11. Verse groups: 2,755 of 20,380 verses (13.5%) share a group. No effect on 10k (2 verses); at full size verse R@10 .522 vs .517 strict.
12. Paid spend, session C: $0.22 ($0.18 questions, $0.04 rewrites). Full-size embedding: ~26 min GPU in 8 parts, lock released between.
