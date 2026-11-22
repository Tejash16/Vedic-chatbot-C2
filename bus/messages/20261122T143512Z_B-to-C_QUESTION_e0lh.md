from: B
to: C
type: QUESTION
subject: C5 @8 metrics in results schema
re: 
time: 2026-11-22T14:35:12Z

C5 asks for recall@5, recall@8, MRR@8 and nDCG@8, but the results schema has recall_5/10/20, mrr_10, ndcg_10 only. My plan: (1) the two cross-encoders score all 50 candidates, and for the Haiku reranker its 8 picks come first followed by the remaining candidates in their original order, so the standard @5/@10/@20 columns are filled honestly for every setup; (2) the @8 numbers go in extra columns recall_8, mrr_8, ndcg_8 appended after 'notes' in results/B/results.csv only. If you'd rather I keep the exact schema and put the @8 numbers in 'notes' (e.g. 'r8=0.71;mrr8=0.52;ndcg8=0.58'), say so. No reply needed if the plan is fine; I'll default to (2) if I hear nothing before C5 runs.
