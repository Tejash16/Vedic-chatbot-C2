"""Write results/A/runs/<run_id>.jsonl: one line per question, {"qid", "chunk_ids": top 50}.
Chunk IDs only (content hashes), no text. C6 ids refer to chunks_150 / chunks_600 / chunks.parquet.

Run from the clone root:  python -m src.A.export_runs
"""
import json

from src.common import load as L
from src.common.score import load_questions

CACHE = L.WORK / "A"
OUT = L.ROOT / "results" / "A" / "runs"
DEPTH = 50


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    qids = list(load_questions().qid)
    written = []
    for p in sorted(CACHE.glob("ranked_C*.json")):
        stage, name = p.stem.split("_", 2)[1:]
        if name == "bm25_ref":                      # reference only, not a result row
            continue
        run_id = name if stage == "C2b" else f"bgem3_ds_{name}"
        ranked = json.loads(p.read_text(encoding="utf-8"))["ranked"]
        assert set(ranked) == set(qids), p.name
        path = OUT / f"{run_id}.jsonl"
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for q in qids:
                f.write(json.dumps({"qid": q, "chunk_ids": ranked[q][:DEPTH]}) + "\n")
        written.append(path)
        print("wrote", path.relative_to(L.ROOT))
    return written


if __name__ == "__main__":
    main()
