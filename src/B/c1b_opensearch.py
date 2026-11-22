"""Step C1b: OpenSearch BM25 arm (session B, laptop 2).

Indexes ../local/work/chunks.parquet (laptop 2's verified chunks; header + text, as
src.common.chunk.embed_text) into a local OpenSearch container and runs every test question:
  os_lang        Hindi-analyzer field (hi/sa chunks) + English-analyzer field (en chunks), best field
  os_plain       unstemmed field (standard tokenizer + lowercase), every chunk
  os_all_mf      one multi_match over the three fields, scores summed (most_fields)
  os_lang+plain  RRF (k = 60, src.common.fusion.rrf) of os_lang and os_plain

Writes stage C1b rows to results/B/results.csv and the top-50 chunk IDs per question to
results/B/runs/<setup>.jsonl, so session C can fuse this arm with others.
Usage (clone root):  python src/B/c1b_opensearch.py [--keep-running]
"""
import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_common as bc  # noqa: E402
import os_index as osi  # noqa: E402

K = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-running", action="store_true", help="leave the container up afterwards")
    a = ap.parse_args()

    if not (bc.WORK / "chunks.verified").exists():
        sys.exit("../local/work/chunks.verified is missing: wait for session A's CHUNKS_VERIFIED")
    chunks = bc.load_chunks()
    qdf = bc.load_questions()
    questions = qdf.to_dict("records")
    print("chunks %d (%s), questions %d" % (len(chunks), dict(chunks.lang.value_counts()), len(questions)))

    client = osi.start_container()
    version = osi.server_version(client)
    t0 = time.perf_counter()
    ok, errors, n = osi.build_index(client, bc.index_rows(chunks))
    index_s = time.perf_counter() - t0
    print("indexed %d docs (bulk ok %d, errors %d) in %.1f s" % (n, ok, len(errors), index_s))
    if errors or n != len(chunks):
        sys.exit("index incomplete: %d of %d" % (n, len(chunks)))

    for q in questions[:20]:  # warm-up, not timed
        for kind in osi.QUERIES:
            osi.search(client, kind, q["question"], K)

    # setups interleaved per question, so every setup sees the same machine load
    runs = {"os_" + k: {} for k in osi.QUERIES}
    lat = {"os_" + k: [] for k in osi.QUERIES}
    for q in questions:
        for kind in osi.QUERIES:
            ids, wall, _took = osi.search(client, kind, q["question"], K)
            runs["os_" + kind][q["qid"]] = ids
            lat["os_" + kind].append(wall)
    name = "os_lang_routed+plain"
    runs[name] = {qid: bc.rrf([runs["os_lang_routed"][qid], runs["os_plain"][qid]], top=K)
                  for qid in runs["os_lang_routed"]}
    lat[name] = [x + y for x, y in zip(lat["os_lang_routed"], lat["os_plain"])]

    common = "OpenSearch %s in Docker (heap %s), BM25 k1=1.2 b=0.75, header+text indexed; latency = client wall clock incl. HTTP, 1 query at a time" % (version, osi.HEAP)
    notes = {
        "os_lang_routed": "Hindi analyzer field holds hi/sa chunks, English analyzer field holds en chunks; question goes to the field of its script (Devanagari -> Hindi, Latin -> English)",
        "os_lang": "NEGATIVE RESULT: both language fields queried, multi_match best_fields (max raw BM25); English questions pulled hi/sa chunks because English words are rare (high IDF) in the Hindi field",
        "os_plain": "unstemmed field: standard tokenizer + lowercase, no stopwords, all chunks",
        "os_all_mf": "multi_match most_fields (scores summed) over Hindi, English and unstemmed fields",
        "os_lang_routed+plain": "RRF k=60 of os_lang_routed and os_plain top-50; latency = sum of both queries",
    }
    rows = []
    for name, run in runs.items():
        bc.save_run(name, run)
        rows += bc.result_rows(run, chunks, session="B", stage="C1b", run_id="C1b_" + name,
                               setup=name, corpus_size=bc.CORPUS_SIZE, latencies_ms=lat[name],
                               paid_usd_per_1000=0.0, notes=notes[name] + "; " + common, questions=qdf)
        for lm in ("lang_match", "chunk_lang"):
            sc = bc.score_run(run, chunks, qdf, by=lm)
            print(name, lm)
            print(sc[sc.question_group != "all"].to_string(index=False))
    bc.write_stage(rows, "C1b")
    bc.print_summary(rows)
    if not a.keep_running:
        osi.stop_container()


if __name__ == "__main__":
    main()
