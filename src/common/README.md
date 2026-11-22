# src/common: shared code (owned by session C)

Run everything from the clone root with `../local/venv` active. `HF_HOME` is set to `../local/hf-cache`
when `src.common.load` is imported. Ask C on the bus for any change here.

## Rebuild the test corpus (one command)

```
python -m src.common.chunk --rebuild
```

Excel (`../local/data`) -> `../local/work/tables/*.parquet` -> `../local/work/records.parquet` ->
`../local/work/chunks.parquet` (10,007 chunks), then checks `shared/chunks_manifest.json` and prints
`MANIFEST MATCH` (exit 0) or `MANIFEST MISMATCH` (exit 1). The manifest lists the SHA-256 of every
Excel file, so a mismatch can be traced to a file. About 2 minutes on CPU.

Other chunk sets from the same records:

| Command | Output | Use |
|---|---|---|
| `python -m src.common.chunk --target 150` | `chunks_150.parquet` | C6 |
| `python -m src.common.chunk --target 600` | `chunks_600.parquet` | C6 (embed with max_length 1024) |
| `python -m src.common.chunk --full` | `chunks_full.parquet` | every hi/sa/en record (C only) |

C6 "300 tokens without header": use `chunks.parquet` with `embed_text(row, header=False)`.

## Chunks

Columns: `chunk_id` (content hash), `record_id` (`<table_id>#<excel row>`), `verse_id` (`<veda>/<ref>`),
`author_id` (the file name as it is: one file = one work), `collection` (rig, sama, yajur, atharva),
`lang` (hi, sa, en: the language of the chunk's columns), `char_start`, `char_end` (span in the record
text), `n_tokens` (bge-m3), `header`, `text`, `sections` (commentary column names the chunk covers).

- Commentary columns per table: `shared/config.yaml`, `tables.<table_id>.commentary` (explicit list).
- Record text: those columns, cleaned (`src.common.text.clean`: NFC, placeholders such as N/A/missing
  removed, `<br>` to line break), in sheet order, joined by a blank line. No column names anywhere.
- Chunks break at a column boundary first, then at sentence ends (danda, ॥, ? !, full stop, line break).
- What to embed: `from src.common.chunk import embed_text; embed_text(row, header=True)` gives
  `header + "\n" + text`. Header: `<Veda> <ref> | <work name>`.

## Questions and scoring

```python
from src.common.chunk import load_chunks
from src.common.score import load_questions, score_run, result_rows, write_results
from src.common.fusion import rrf          # reciprocal rank fusion, k = 60
from src.common.text import words          # the C1a tokenizer (split on spaces, punctuation, danda)

chunks = load_chunks()                     # or load_chunks(150), load_chunks(600)
q = load_questions()                       # shared/test_questions.jsonl, 360 rows
ranked = {qid: [chunk_id, ...]}            # best first, 20 or more per question
print(score_run(ranked, chunks))           # per group and 'all', chunk and verse level
print(score_run(ranked, chunks, by="lang_match"))   # same- vs cross-language
rows = result_rows(ranked, chunks, session="A", stage="C2b", run_id="qwen_dense",
                   setup="Qwen3-Embedding-0.6B dense", corpus_size="10k",
                   latencies_ms=lat, paid_usd_per_1000=0.0, notes="")
write_results(rows, "results/A/results.csv")
```

- Question fields: `qid, group, question, gold_chunk_id, record_id, verse_id, char_start, char_end,
  chunk_lang, q_lang, style, verse_group, collection, author_id`. Groups: `hi`, `en`, `hi_roman`,
  `sa_quote` (90 each). Each group has 36 Hindi, 32 English and 22 Sanskrit gold chunks.
- Chunk level: the gold chunk. For C6 chunk sizes pass `overlap_gold=True`: any chunk of the gold record
  overlapping the gold span counts.
- Verse level: any chunk of any verse in the gold verse's `verse_group` (the same mantra in several Vedas).
- B (C5): pass `at8=True` to add `recall_8, mrr_8, ndcg_8` after `notes`.
- A group `real` can be appended later to `shared/test_questions.jsonl`; it is reported on its own
  and never folded into `all`.

## Verses

`from src.common.verses import load_verses` -> `../local/work/verses.parquet`: `verse_id, collection,
ref, verse_text, verse_text_accented, pada, pada_accented, devata, rishi, chhanda, swara, verse_group,
group_size` (all cleaned the same way as commentary).

## bge-m3 on Windows (tip from D)

`BGEM3FlagModel('BAAI/bge-m3')` may try to download extra repo files and fail on a symlink. Load the local
snapshot instead: `path = snapshot_download('BAAI/bge-m3', local_files_only=True)` with `HF_HUB_OFFLINE=1`,
then `BGEM3FlagModel(path, use_fp16=True)`.
