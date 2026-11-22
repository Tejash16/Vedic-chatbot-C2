# C2 benchmark specification

A small, reproducible benchmark on our own data that decides the open choices in the retrieval design. Accuracy matters most, then latency, then cost.

Four Claude Code sessions run it on two laptops. Your own tasks are in `roles/<your letter>.md`. The coordination rules are in `PROTOCOL.md`.

## What we are building

A question-answering assistant over a library of short verses (mantras), each with several commentaries (translations) by different authors, in Sanskrit, Hindi and English. Users ask in any language, often Hindi typed in Roman letters. Answers must come only from retrieved passages, with exact references.

## Rules for every session

- **Folders.** Your clone is the Git repo. `../local` is this laptop's private folder, shared by the two sessions on it: `../local/data` (Excel files, never modified), `../local/work` (derived files), `../local/final` (reports), `../local/hf-cache` (models), `../local/venv`, `../local/.env`. Use the Python environment in `../local/venv`, and set `HF_HOME` to `../local/hf-cache` so models are not downloaded twice.
- **Source text never goes to Git.** No Excel, Parquet, chunks, embeddings, passages, audit samples or failure cases. Git carries only code, config, test questions, chunk IDs, metrics and bus messages.
- **Paid APIs: Anthropic only.** The key is in `../local/.env` as `VEDIC_ANTHROPIC_API_KEY`. Pass it to the client explicitly. Never print it. Never create or export a variable named `ANTHROPIC_API_KEY`.
- **Models.** `claude-haiku-5-5` for bulk work. `claude-sonnet-5-5` only where stated. If a model ID fails, check Anthropic's current model list.
- **Spending limits in US dollars.** C: 5. D: 10. B: 5. A: 0. Before each paid step show the human the estimated cost and wait for a yes. Log every paid step in `results/<letter>/spend.log`.
- **GPU.** One GPU job per laptop at a time. Take the lock before and release it after (see `PROTOCOL.md`). Use half precision. Set the maximum length explicitly: 512 tokens for passages unless stated.
- **Reproducible.** Random seed 7 everywhere. Chunk IDs come from a content hash.
- **Simple.** Python, no frameworks unless needed. Cache embeddings in `../local/work`.
- **Check the machine first** (RAM, CPU, GPU memory, free disk, Docker) and size batches to fit. Tell the human the expected run time before each long step.

## Shared files and their owners

| File | In Git | Written by | Holds |
|---|---|---|---|
| `../local/final/data-audit.md` | No | C | Stage A report |
| `../local/work/tables/*.parquet` | No | C on laptop 1, A on laptop 2 | Excel converted |
| `../local/work/chunks.parquet` | No | C on laptop 1, A on laptop 2 | chunk_id, record_id, verse_id, author_id, collection, lang, char_start, char_end, n_tokens, header, text |
| `../local/work/chunks.verified` | No | A | Marker: laptop 2's chunks match the manifest |
| `shared/chunks_manifest.json` | Yes | C | Chunk count and a SHA-256 over the sorted chunk IDs |
| `shared/test_questions.jsonl` | Yes | C | qid, group, question, gold_chunk_id, record_id, verse_id, char_start, char_end |
| `shared/config.yaml` | Yes | C | Every candidate and setting |
| `src/common/` | Yes | C only | Loading, chunking, scoring and the results writer |
| `src/<letter>/` | Yes | That session | Its own stage code |
| `shared/candidates_top50.jsonl` | Yes | C | qid and the top 50 chunk IDs from the best fused setup |
| `results/<letter>/results.csv` | Yes | That session | Its measurements, in the shared schema |
| `results/<letter>/findings.md` | Yes | That session | What it found, in 10 to 20 lines, with no source text |

**Results schema** (one row per run and question group): session, stage, run_id, setup, corpus_size, question_group, level (chunk or verse), recall_5, recall_10, recall_20, mrr_10, ndcg_10, p50_ms, p95_ms, paid_usd_per_1000, machine, notes.

## Stage A. Data audit (session C, no paid calls)

1. Load every file and sheet in `../local/data`. Use a reader that handles large files, and convert to Parquet.
2. Report for each table: rows, columns, what each column appears to hold, sample values, languages and scripts, empty and duplicate rates.
3. Work out how the tables link: verse to commentary to author. Report broken links, and counts per author and per collection.
4. Check for Excel damage: cells cut at 32,767 characters, verse numbers turned into decimals or dates (for example 1.10 stored as 1.1), formulas or error values, mixed Unicode forms, stray zero-width characters, HTML tags.
5. Report text lengths per author: characters, words and sentences per record, with the distribution. Split Hindi and Sanskrit sentences on the danda. Estimate tokens by counting 200 sampled records with Anthropic's token counting and scaling up.
6. Say whether a word-by-word (pada) form of each verse is present.
7. Write `../local/final/data-audit.md`. Then stop, show the human the summary and your assumptions about the columns, and wait for a go.

## Stage B. Chunks and test set (session C)

1. Normalize text to NFC. Split commentaries into chunks of about 300 tokens (bge-m3 tokenizer) at sentence ends. The danda counts as a sentence end. Keep record, verse and author IDs and the character span. Put a short header on each chunk: verse reference, author, collection.
2. Build a test corpus of about 10,000 chunks spread across authors, languages and lengths. The choice must be deterministic so laptop 2 rebuilds the same chunks from the same Excel files.
3. With `claude-haiku-5-5`, write 360 questions, each tied to one chunk that answers it: 90 in Hindi, 90 in English, 90 in Hindi typed in Roman letters the way people type on a phone, and 90 that quote a few Sanskrit words from the verse. Mix direct, reworded and broad questions. A question must not copy the chunk's wording.
4. If the human gives a file of real questions, add them as the group `real`. Always report generated and real questions separately.
5. Write the shared code in `src/common`, the manifest, the questions and `config.yaml`. Push them and announce `B_READY`.

## Stage C. Retrieval

| Step | What | Session |
|---|---|---|
| C1a | BM25 with our own tokenizer: split on spaces, punctuation and danda after NFC, never inside a word | C |
| C1b | OpenSearch BM25 with its Hindi analyzer on Hindi and Sanskrit text and the English analyzer on English text, plus an unstemmed field. Only if Docker is available | B |
| C2a | `BAAI/bge-m3` dense, sparse, and dense plus sparse | C |
| C2b | `Qwen/Qwen3-Embedding-0.6B` dense, alone and fused with C1a | A |
| C3 | Fusion is reciprocal rank fusion with k = 60 everywhere | All |
| C4 | Question forms: as typed. Rewritten by `claude-haiku-5-5` into a Devanagari form and an English form. Both fused with the original | C |
| C5 | Rerankers on `shared/candidates_top50.jsonl`, keeping 8: `BAAI/bge-reranker-v2-m3`, `Qwen/Qwen3-Reranker-0.6B`, and `claude-haiku-5-5` choosing the best 8. Report recall at 5 and 8, MRR and nDCG at 8, and the same numbers with no reranker | B |
| C6 | With bge-m3 dense plus sparse: chunks of about 150 tokens, about 600 tokens, and 300 tokens with the header removed. The right answer is any chunk of the same record that overlaps the gold character span | A |

Measure for every run: recall at 5, 10 and 20, MRR and nDCG at 10, split by question group. Report recall at chunk level and at verse level (any chunk of the right verse counts). Latency at p50 and p95 for each stage on that machine. Paid cost per 1,000 questions.

## Stage D. Question-and-answer sets (session D)

Use a sub-corpus of about 1,500 chunks that holds at least 150 of the test questions. Generate sets for every chunk in it, not only for the answer chunks.

1. Method, with `claude-haiku-5-5`: first list every distinct point the passage makes, each as an exact quote from the passage. Then write one set per point: a question in Hindi, the same question in English, and the quote as the answer. There is no fixed number of sets.
2. Filters, all in code: drop a set if its answer is not found word for word in the passage after normalization. Drop it if a name or Sanskrit term in the question appears nowhere in the passage or the verse details. Drop it if the question does not bring back its own passage in the top 10. Merge repeats.
3. Report: sets per passage for short, medium and long passages, sets per 100 words, the share removed by each filter, and the share of each passage's sentences covered by at least one set.
4. Scale up: from sets per 100 words and the audit, estimate the total number of sets and the `claude-haiku-5-5` Batch API cost, for each author and for the whole dataset. Write `../local/final/sets-estimate.md`.
5. Search test with bge-m3: passages only, against passages plus all sets, against passages plus the best 5, 10, 20 and 40 sets per passage. Also compare indexing the source-language question only against both languages. Each question vector points to its passage, and results are merged by passage.
6. On 100 passages, also generate sets with `claude-sonnet-5-5`. Write 60 sets from each model to `../local/final/sets-for-review.csv`, in random order with the model name hidden and the passage beside each set, for a scholar to mark right or wrong. Keep the answer key in a separate file beside it.

## Final results (session C only)

When A, B and D have each announced `DONE`:

1. Pull. Merge every `results/<letter>/results.csv` into `results/all_results.csv`.
2. Ask any session whose `findings.md` is missing or unclear, over the bus.
3. Write `../local/final/benchmark-report.md`: a results table, a ranked recommendation for each choice (keyword arm, embedding model, question form, reranker, chunk size, header, sets to index), 20 failure cases where every setup missed, what to test next, and the two machines it ran on.
4. Write `results/SUMMARY.md` for Git: the same table and recommendations, with no source text.
5. Announce `DONE` to ALL and tell the human which files to send on.
