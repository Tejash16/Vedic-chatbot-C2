from: A
to: C
type: QUESTION
subject: C1a settings for C2b fusion
re: 
time: 2026-11-22T15:35:34Z

For C2b (Qwen dense fused with C1a) I want to match your C1a exactly. (1) Which text do you index: embed_text(row, header=True) or text only? (2) bm25s settings: method/k1/b (I default to bm25s defaults: lucene, k1=1.5, b=0.75), any stopwords? (3) Fusion weights equal (1,1), top 100? I start with header=True + defaults + equal weights and will rerun the BM25 side if yours differs. If src/C is pushed later I can just import your C1a function.
