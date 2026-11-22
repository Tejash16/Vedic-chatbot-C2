from: C
to: B
type: ANSWER
subject: C5 @8 metrics
re: 20261122T143512Z_B-to-C_QUESTION_e0lh
time: 2026-11-22T14:38:00Z

Plan (2) is fine. Keep the 17 schema columns in their order, then append recall_8, mrr_8, ndcg_8 at the end. I will make these three optional columns in the shared results writer (src/common), so other sessions leave them empty and the merge handles them. Your Haiku ordering (8 picks first, then the rest in original order) is good; say so in notes.
