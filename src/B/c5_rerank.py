"""Step C5: rerankers on shared/candidates_top50.jsonl, keeping 8 (session B).

  python src/B/c5_rerank.py bge          # GPU: BAAI/bge-reranker-v2-m3
  python src/B/c5_rerank.py qwen         # GPU: Qwen/Qwen3-Reranker-0.6B
  python src/B/c5_rerank.py estimate     # free: count Haiku input tokens on a sample, print cost
  python src/B/c5_rerank.py haiku        # PAID: claude-haiku-5-5 picks the best 8 (resumable)
  python src/B/c5_rerank.py score        # metrics for every finished run -> results/B/results.csv

Runs (chunk IDs, latencies, token counts; no source text) go to ../local/work/B/c5/.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import b_common as bc  # noqa: E402
import rerank_models as rm  # noqa: E402

OUT = bc.WORK / "B" / "c5"
HAIKU_IN_USD, HAIKU_OUT_USD = 0.10 / 1e6, 0.50 / 1e6  # claude-haiku-5-5, prompts up to 100K tokens


def load_candidates():
    cands = {}
    with open(bc.SHARED / "candidates_top50.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            ids = d.get("chunk_ids") or d.get("candidates") or d.get("top50") or d.get("ids")
            cands[d["qid"]] = list(ids)
    return cands


def setup():
    chunks = bc.load_chunks()
    questions = bc.questions_list()
    cands = load_candidates()
    text_of = {r["chunk_id"]: bc.index_text(r) for r in chunks.to_dict("records")}
    missing = {c for ids in cands.values() for c in ids if c not in text_of}
    if missing:
        sys.exit("%d candidate chunk IDs are not in chunks.parquet" % len(missing))
    lost = [q["qid"] for q in questions if q["qid"] not in cands]
    if lost:
        sys.exit("%d questions have no candidates" % len(lost))
    return chunks, questions, cands, text_of


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ("%s.json" % name)).write_text(json.dumps(data), encoding="utf-8")


def run_local(kind):
    chunks, questions, cands, text_of = setup()
    model = rm.BgeReranker() if kind == "bge" else rm.QwenReranker()
    for q in questions[:3]:  # warm-up
        model.rank(q["question"], [text_of[c] for c in cands[q["qid"]]])
    run, lat = {}, {}
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        ids = cands[q["qid"]]
        order, ms, _ = model.rank(q["question"], [text_of[c] for c in ids])
        run[q["qid"]] = [ids[j] for j in order]
        lat[q["qid"]] = ms
        if i % 50 == 0:
            print("%s %d/%d  %.0f s" % (kind, i, len(questions), time.time() - t0), flush=True)
    model.close()
    save(kind, {"model": model.name, "run": run, "latency_ms": lat})
    print("done %s in %.0f s" % (kind, time.time() - t0))


def client():
    import anthropic
    from dotenv import dotenv_values
    key = dotenv_values(bc.LOCAL / ".env").get("VEDIC_ANTHROPIC_API_KEY")
    if not key:
        sys.exit("VEDIC_ANTHROPIC_API_KEY missing from ../local/.env")
    return anthropic.Anthropic(api_key=key, max_retries=4, timeout=180)


def estimate(sample=24, effort="medium"):
    chunks, questions, cands, text_of = setup()
    c = client()
    step = max(1, len(questions) // sample)
    picked = questions[::step][:sample]
    toks = []
    for q in picked:
        req = rm.haiku_request(q["question"], [text_of[x] for x in cands[q["qid"]]], effort=effort)
        r = c.messages.count_tokens(model=req["model"], system=req["system"], messages=req["messages"],
                                    output_config=req["output_config"])
        toks.append(r.input_tokens)
    mean_in, max_in = sum(toks) / len(toks), max(toks)
    n = len(questions)
    print("sample %d questions: mean input %.0f tokens, max %d" % (len(toks), mean_in, max_in))
    for out_tok in (300, 1500, 3000):
        usd = n * (mean_in * HAIKU_IN_USD + out_tok * HAIKU_OUT_USD)
        print("  %d questions, %d output tokens each: $%.2f" % (n, out_tok, usd))
    return mean_in, max_in


def log_spend(step, n_req, tin, tout, usd):
    bc.RES.mkdir(parents=True, exist_ok=True)
    with open(bc.RES / "spend.log", "a", encoding="utf-8") as f:
        f.write("%s\t%s\t%s\trequests=%d\tinput_tokens=%d\toutput_tokens=%d\tusd=%.4f\n" % (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            step, rm.HAIKU_MODEL, n_req, tin, tout, usd))


def run_haiku(effort="medium", workers=4, limit_usd=4.5):
    chunks, questions, cands, text_of = setup()
    c = client()
    name = "haiku_%s" % effort
    raw_path = OUT / ("%s_raw.jsonl" % name)
    OUT.mkdir(parents=True, exist_ok=True)
    done = {}
    if raw_path.exists():
        for line in raw_path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done[d["qid"]] = d
    todo = [q for q in questions if q["qid"] not in done]
    print("%s: %d done, %d to do" % (name, len(done), len(todo)))
    lock = threading.Lock()
    spent = {"usd": sum(d["usd"] for d in done.values()), "in": 0, "out": 0, "n": 0}

    def one(q):
        ids = cands[q["qid"]]
        req = rm.haiku_request(q["question"], [text_of[x] for x in ids], effort=effort)
        t0 = time.perf_counter()
        msg = c.messages.create(**req)
        ms = (time.perf_counter() - t0) * 1000.0
        picks, status = rm.haiku_parse(msg, len(ids))
        u = msg.usage
        usd = u.input_tokens * HAIKU_IN_USD + u.output_tokens * HAIKU_OUT_USD
        return {"qid": q["qid"], "picks": picks, "status": status, "ms": ms,
                "input_tokens": u.input_tokens, "output_tokens": u.output_tokens, "usd": usd,
                "stop_reason": msg.stop_reason}

    try:
        with cf.ThreadPoolExecutor(workers) as ex, open(raw_path, "a", encoding="utf-8") as f:
            futs = {}
            it = iter(todo)
            for q in [next(it, None) for _ in range(workers)]:
                if q:
                    futs[ex.submit(one, q)] = q
            while futs:
                fin, _ = cf.wait(futs, return_when=cf.FIRST_COMPLETED)
                for fu in fin:
                    q = futs.pop(fu)
                    try:
                        d = fu.result()
                    except Exception as e:  # keep going; the question is retried on the next run
                        print("ERROR %s: %r" % (q["qid"], e)[:300], flush=True)
                        continue
                    with lock:
                        f.write(json.dumps(d) + "\n")
                        f.flush()
                        done[d["qid"]] = d
                        spent["usd"] += d["usd"]
                        spent["in"] += d["input_tokens"]
                        spent["out"] += d["output_tokens"]
                        spent["n"] += 1
                        if spent["n"] % 25 == 0:
                            print("%d/%d  spent so far $%.3f" % (len(done), len(questions), spent["usd"]), flush=True)
                    if spent["usd"] < limit_usd:
                        nq = next(it, None)
                        if nq:
                            futs[ex.submit(one, nq)] = nq
                    else:
                        print("spend limit $%.2f reached, stopping" % limit_usd)
    finally:
        if spent["n"]:
            log_spend("C5 Haiku reranker (%s effort), real-time Messages API" % effort,
                      spent["n"], spent["in"], spent["out"],
                      spent["in"] * HAIKU_IN_USD + spent["out"] * HAIKU_OUT_USD)
    bad = [d for d in done.values() if d["status"] != "ok"]
    print("finished %d/%d, not ok: %d, total $%.3f" % (len(done), len(questions), len(bad),
                                                       sum(d["usd"] for d in done.values())))
    run = {qid: [cands[qid][j] for j in rm.haiku_order(d["picks"], len(cands[qid]))] for qid, d in done.items()}
    save(name, {"model": rm.HAIKU_MODEL, "effort": effort, "run": run,
                "latency_ms": {qid: d["ms"] for qid, d in done.items()},
                "usd": {qid: d["usd"] for qid, d in done.items()},
                "status": {qid: d["status"] for qid, d in done.items()}})


def score():
    chunks, questions, cands, text_of = setup()
    qdf = bc.load_questions()
    rows = []
    rows += bc.result_rows(cands, chunks, session="B", stage="C5", run_id="C5_no_reranker",
                           setup="no_reranker", corpus_size=bc.CORPUS_SIZE,
                           latencies_ms=[0.0] * len(questions), questions=qdf, at8=True,
                           notes="candidates_top50 order as given; no rerank step, so no added latency")
    notes = {
        "bge": "BAAI/bge-reranker-v2-m3 fp16, pair max 512 tokens (passage truncated), scores all 50; latency = 50 pairs per question on RTX 3050 4GB",
        "qwen": "Qwen/Qwen3-Reranker-0.6B fp16, yes/no last-token logit, passage cut at 512 tokens, task instruction; scores all 50; latency = 50 pairs per question on RTX 3050 4GB",
        "haiku_medium": "claude-haiku-5-5 effort medium, structured JSON output, sees 50 numbered passages in candidate order; its 8 picks first, then the other candidates in original order; latency = one real-time API call",
        "haiku_low": "claude-haiku-5-5 effort low, otherwise as haiku_medium",
    }
    for name in ("bge", "qwen", "haiku_medium", "haiku_low"):
        p = OUT / ("%s.json" % name)
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if len(d["run"]) != len(questions):
            print("WARNING %s covers %d of %d questions; missing ones are scored as misses"
                  % (name, len(d["run"]), len(questions)))
        paid = 0.0
        if "usd" in d:
            paid = 1000.0 * sum(d["usd"].values()) / max(1, len(d["usd"]))
        setup_name = {"bge": "bge-reranker-v2-m3", "qwen": "qwen3-reranker-0.6b"}.get(name, "claude-" + name.replace("_", "-"))
        extra = ""
        if "status" in d:
            nb = sum(1 for s in d["status"].values() if s != "ok")
            extra = "; %d answers not ok (refusal/short/bad json)" % nb
        rows += bc.result_rows(d["run"], chunks, session="B", stage="C5", run_id="C5_" + name,
                               setup=setup_name, corpus_size=bc.CORPUS_SIZE,
                               latencies_ms=list(d["latency_ms"].values()), paid_usd_per_1000=paid,
                               questions=qdf, at8=True, notes=notes[name] + extra)
        for lm in ("lang_match", "chunk_lang"):
            sc = bc.score_run(d["run"], chunks, qdf, by=lm, at8=True)
            print(name, lm)
            print(sc[sc.question_group != "all"].to_string(index=False))
    bc.write_stage(rows, "C5")
    bc.print_summary(rows, at8=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bge", "qwen", "estimate", "haiku", "score"])
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-usd", type=float, default=4.5)
    a = ap.parse_args()
    if a.cmd in ("bge", "qwen"):
        run_local(a.cmd)
    elif a.cmd == "estimate":
        estimate(effort=a.effort)
    elif a.cmd == "haiku":
        run_haiku(effort=a.effort, workers=a.workers, limit_usd=a.limit_usd)
    else:
        score()


if __name__ == "__main__":
    main()
