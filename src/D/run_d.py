"""Stage D command line. Run from the clone root with the laptop venv active.

  python src/D/run_d.py dryrun                     # whole pipeline on made-up passages, no paid calls
  python src/D/run_d.py select                     # build the ~1,500-chunk sub-corpus
  python src/D/run_d.py estimate --model haiku     # cost estimate (token counting, free)
  python src/D/run_d.py pilot --n 10               # paid: a few sync calls to check the prompt
  python src/D/run_d.py generate --model haiku     # paid: Batch API over the sub-corpus
  python src/D/run_d.py generate --model sonnet    # paid: Batch API over the 100 review passages
  python src/D/run_d.py filter --model haiku       # GPU: filters 1-3 and merge
  python src/D/run_d.py report                     # step 3 statistics
  python src/D/run_d.py scale                      # step 4 estimate -> ../local/final/sets-estimate.md
  python src/D/run_d.py search                     # GPU: step 5 search test
  python src/D/run_d.py review                     # step 6 -> ../local/final/sets-for-review.csv
  python src/D/run_d.py results                    # results/D/results.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data  # noqa: E402
import filters  # noqa: E402
import gen  # noqa: E402
import stage  # noqa: E402
from dutil import (FINAL, HAIKU, RESULTS_D, SEED, SONNET, WORK_D, machine_string, n_words, read_jsonl,  # noqa: E402
                   utf8_stdout, write_jsonl)
from embed import PASSAGE_MAXLEN, QUERY_MAXLEN, FakeEmbedder, LazyBGEM3, cached_encode  # noqa: E402
from terms import TermChecker  # noqa: E402

MODELS = {"haiku": HAIKU, "sonnet": SONNET}
N_REVIEW_PASSAGES = 100


class Ctx:
    def __init__(self, dry=False, base=None):
        self.dry = dry
        if dry:
            base = Path(base)
            self.work, self.final, self.results = base / "work", base / "final", base / "results"
        else:
            self.work, self.final, self.results = WORK_D, FINAL, RESULTS_D
        for d in (self.work, self.final, self.results):
            d.mkdir(parents=True, exist_ok=True)

    # ---- inputs
    def passages(self):
        if self.dry:
            return data.fake_passages()
        return read_jsonl(self.work / "subcorpus.jsonl")

    def questions(self):
        return data.fake_questions() if self.dry else data.load_questions()

    def lexicon(self):
        return data.fake_lexicon() if self.dry else data.load_lexicon(self.work)

    def embedder(self):
        return FakeEmbedder() if self.dry else LazyBGEM3()

    def index_text(self, p):
        return data.index_text(p) if not self.dry else "%s\n%s" % (p["header"], p["text"])

    def review_ids(self, passages):
        """100 passages for the Sonnet comparison: stratified by length bin, seed 7."""
        t1, t2 = stage.length_bins(passages)
        bins = {"short": [], "medium": [], "long": []}
        for p in sorted(passages, key=lambda p: p["chunk_id"]):
            bins[stage.bin_of(n_words(p["text"]), t1, t2)].append(p["chunk_id"])
        rnd = random.Random(SEED)
        n = min(N_REVIEW_PASSAGES, len(passages))
        out = []
        for i, (k, v) in enumerate(sorted(bins.items())):
            take = n // 3 + (1 if i < n % 3 else 0)
            out.extend(rnd.sample(v, min(take, len(v))))
        return sorted(out)

    def gen_path(self, model):
        return self.work / ("gen_%s.json" % model.split("-")[1])

    def sets_path(self, model):
        return self.work / ("sets_%s.jsonl" % model.split("-")[1])


# ------------------------------------------------------------------ steps

def do_generate(ctx, model, use_batch=True):
    passages = ctx.passages()
    if model == SONNET:
        ids = set(ctx.review_ids(passages))
        passages = [p for p in passages if p["chunk_id"] in ids]
    if ctx.dry:
        res = gen.fake_generate(passages)
    else:
        res = run_gen(ctx, model, passages, use_batch, "D1 generate %s" % model, tag="")
        # one retry for passages whose answer had fewer sets than points, or failed
        short = [p for p in passages if res[p["chunk_id"]].get("error")
                 or len(res[p["chunk_id"]]["sets"]) < len(res[p["chunk_id"]]["points"])]
        if short:
            print("retrying %d passages with fewer sets than points or an error" % len(short))
            res2 = run_gen(ctx, model, short, use_batch, "D1 retry %s" % model, tag="_retry")
            for p in short:
                a, b = res[p["chunk_id"]], res2[p["chunk_id"]]
                better = (not b.get("error")) and (a.get("error") or
                                                   len(b["sets"]) - len(b["points"]) > len(a["sets"]) - len(a["points"]))
                keep = dict(b if better else a)
                keep["in_tokens"] = a["in_tokens"] + b["in_tokens"]      # both attempts are paid
                keep["out_tokens"] = a["out_tokens"] + b["out_tokens"]
                keep["retried"] = True
                res[p["chunk_id"]] = keep
    ctx.gen_path(model).write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    errs = sum(1 for r in res.values() if r.get("error"))
    print("generated for %d passages, %d with errors" % (len(res), errs))
    return res


def run_gen(ctx, model, passages, use_batch, step, tag=""):
    if not use_batch:
        return gen.run_sync(model, passages, step + " sync")
    bid, id_map = gen.submit_batch(model, passages)
    (ctx.work / ("batch_%s%s.json" % (model.split("-")[1], tag))).write_text(
        json.dumps({"batch_id": bid, "id_map": id_map}), encoding="utf-8")
    print("submitted batch", bid, "with", len(passages), "requests", flush=True)
    gen.wait_batch(bid, log=lambda s: print(s, flush=True))
    return gen.collect_batch(bid, model, id_map, step + " batch")


def do_filter(ctx, model, emb=None):
    passages = ctx.passages()
    pmap = {p["chunk_id"]: p for p in passages}
    g = json.loads(ctx.gen_path(model).read_text(encoding="utf-8"))
    sets, bstats = filters.build_sets(g, pmap, model)
    checker = TermChecker(*ctx.lexicon())
    own = emb is None
    emb = emb or ctx.embedder()
    try:
        P = cached_encode(emb, [ctx.index_text(p) for p in passages], PASSAGE_MAXLEN, "passages",
                          ctx.work / "emb")
        Qhi = cached_encode(emb, [s["question_hi"] for s in sets], QUERY_MAXLEN, "q_hi_" + model, ctx.work / "emb")
        Qen = cached_encode(emb, [s["question_en"] for s in sets], QUERY_MAXLEN, "q_en_" + model, ctx.work / "emb")
    finally:
        if own:
            emb.close()
    pid_index = {p["chunk_id"]: i for i, p in enumerate(passages)}
    filters.apply_all(sets, pmap, checker, pid_index, P, Qhi, Qen)
    write_jsonl(ctx.sets_path(model), sets)
    rep = filters.filter_report(sets)
    rep["build"] = bstats
    (ctx.work / ("filter_%s.json" % model.split("-")[1])).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps(rep, indent=1))
    return sets


def do_report(ctx):
    passages = ctx.passages()
    out = {}
    sh = read_jsonl(ctx.sets_path(HAIKU))
    out["haiku_all"] = stage.stats(passages, sh)
    out["haiku_filters"] = json.loads((ctx.work / "filter_haiku.json").read_text(encoding="utf-8"))
    if ctx.sets_path(SONNET).exists():
        ids = set(ctx.review_ids(passages))
        sub = [p for p in passages if p["chunk_id"] in ids]
        ss = read_jsonl(ctx.sets_path(SONNET))
        out["review100_haiku"] = stage.stats(sub, [s for s in sh if s["chunk_id"] in ids])
        out["review100_sonnet"] = stage.stats(sub, ss)
        out["review100_haiku_filters"] = filters.filter_report([s for s in sh if s["chunk_id"] in ids])
        out["review100_sonnet_filters"] = filters.filter_report(ss)
    (ctx.work / "report.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    a = out["haiku_all"]
    for k in ("short", "medium", "long", "all"):
        if k in a:
            r = a[k]
            print("%-7s passages=%4d words/p=%6.1f gen/p=%5.2f kept/p=%5.2f kept/100w=%5.2f cov=%s" % (
                k, r["passages"], r["words_per_passage"], r["sets_generated_per_passage"],
                r["sets_kept_per_passage"], r["sets_kept_per_100_words"],
                "%.2f" % r["sentence_coverage_kept"] if r["sentence_coverage_kept"] is not None else "-"))
    print("bins:", a["_bins"])
    return out


def do_scale(ctx):
    passages = ctx.passages()
    sets = read_jsonl(ctx.sets_path(HAIKU))
    g = json.loads(ctx.gen_path(HAIKU).read_text(encoding="utf-8"))
    authors = data.fake_authors() if ctx.dry else data.load_author_totals(ctx.work)
    rows, fits = stage.scale_up(authors, passages, sets, g)
    rows2, _ = stage.scale_up(authors, passages, sets, g, kept_key="kept_nof3")
    for r, r2 in zip(rows, rows2):
        r["sets_per_100_words_nof3"] = r2["sets_per_100_words"]
        r["est_sets_nof3"] = r2["est_sets"]
    (ctx.work / "scale.json").write_text(json.dumps({"rows": rows, "fits": fits}, indent=1, ensure_ascii=False),
                                         encoding="utf-8")
    write_estimate_md(ctx, rows, fits)
    return rows, fits


def write_estimate_md(ctx, rows, fits):
    tot_sets = sum(r["est_sets"] for r in rows)
    tot_sets2 = sum(r.get("est_sets_nof3", 0) for r in rows)
    tot_usd = sum(r["est_batch_usd"] for r in rows)
    tot_words = sum(r["words"] for r in rows)
    tot_req = sum(r["est_requests"] for r in rows)
    by_lang = {}
    for r in rows:
        d = by_lang.setdefault(r.get("lang", "?"), [0, 0, 0.0, 0.0, 0.0])
        d[0] += r["words"]
        d[1] += r["est_requests"]
        d[2] += r["est_sets"]
        d[3] += r["est_batch_usd"]
        d[4] += r.get("est_sets_nof3", 0)
    L = ["# Question-and-answer sets: scale-up estimate (session D)", "",
         "Model: `claude-haiku-5-5`, Message Batches API (50 % of the standard $0.10 / $0.50 per million "
         "input / output tokens). One request per chunk of about 300 bge-m3 tokens, as in Stage B.", "",
         "Method. Sets per 100 words are measured on the Stage D sub-corpus (1,499 chunks; kept sets after "
         "all filters and merging). Words, records and characters per work come from "
         "`../local/work/records.parquet` (every language). Requests per work = sum over records of "
         "ceil(bge-m3 tokens / 300), with tokens from the audit's tokens-per-character ratios; this formula "
         "gives 9,915 chunks for the 10k corpus records, against 10,007 real ones. Token use per request is "
         "fitted on the real batch requests as tokens = a + b * x, where x is the passage size in Claude "
         "tokens (characters times the audit's Claude tokens per character for the language), so the same "
         "model carries over to languages that are not in the sub-corpus.", "",
         "Two set counts are given: after all three filters and merging (the spec's definition), and "
         "without filter 3 (filters 1 and 2 and merging only). The search test found the second index "
         "better (chunk recall@10 0.61 against 0.50 and 0.47 for passages alone), so it is the count to plan "
         "with. The generation cost is the same for both.", "",
         "## Whole dataset", "",
         "| Words | Requests | Sets, all filters | Sets, without filter 3 | Estimated batch cost (USD) |",
         "|---:|---:|---:|---:|---:|",
         "| %s | %s | %s | %s | %.2f |" % (f"{tot_words:,}", f"{tot_req:,}", f"{round(tot_sets):,}",
                                        f"{round(tot_sets2):,}", tot_usd), "",
         "## By language", "",
         "| Language | Words | Requests | Sets, all filters | Sets, without filter 3 | Batch USD |",
         "|---|---:|---:|---:|---:|---:|"]
    for lang, d in sorted(by_lang.items(), key=lambda kv: -kv[1][0]):
        L.append("| %s | %s | %s | %s | %s | %.2f |" % (lang, f"{d[0]:,}", f"{round(d[1]):,}", f"{round(d[2]):,}",
                                                    f"{round(d[4]):,}", d[3]))
    L += ["", "## Token model (per request)", "",
          "| Fit | Requests measured | Input a | Input b per passage token | Output a | Output b per passage token |",
          "|---|---:|---:|---:|---:|---:|"]
    for sc, f in sorted(fits.items()):
        L.append("| %s | %d | %.0f | %.2f | %.0f | %.2f |" % (sc, f["n"], f["in_a"], f["in_b"], f["out_a"], f["out_b"]))
    L += ["", "The estimate uses the `all` fit for every work.", "", "## Per author (work)", "",
          "| Author (work) | Language | Words | Requests | Sets / 100 words (all filters / no F3) | Rate from "
          "| Sets, all filters | Sets, no F3 | Batch USD |",
          "|---|---|---:|---:|---:|---|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: -r["words"]):
        L.append("| %s | %s | %s | %s | %.2f / %.2f | %s | %s | %s | %.3f |" % (
            r.get("author") or r["author_id"], r.get("lang", ""), f"{r['words']:,}", f"{round(r['est_requests']):,}",
            r["sets_per_100_words"], r.get("sets_per_100_words_nof3", 0), r["rate_source"],
            f"{round(r['est_sets']):,}", f"{round(r.get('est_sets_nof3', 0)):,}", r["est_batch_usd"]))
    L += ["", "Notes:",
          "- A rate marked `script:` or `overall` is borrowed "
          "because the sub-corpus held fewer than 15 passages of that work; works in Bengali, Gujarati, "
          "Marathi, Tamil, Urdu, Nepali and Hinglish are not in the sub-corpus, so their set counts are the "
          "least certain (sets per 100 words differ with word length).",
          "- Retries are included: in the sub-corpus run 24 of 1,499 passages (1.6 %) were sent again because "
          "the answer had fewer sets than points, and both attempts' tokens are inside the measured fit.",
          "- Filter 3 needs a bge-m3 pass over every question and passage (GPU time, no fee).",
          "- About half of each request's input is the fixed instructions; sending several passages per "
          "request, or prompt caching once the fixed part passes the model's minimum cacheable length, "
          "would cut input cost further."]
    (ctx.final / "sets-estimate.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", ctx.final / "sets-estimate.md", "| sets all filters %.0f | sets no F3 %.0f | batch USD %.2f" % (
        tot_sets, tot_sets2, tot_usd))


def do_search(ctx, emb=None):
    passages = ctx.passages()
    sets = read_jsonl(ctx.sets_path(HAIKU))
    questions = ctx.questions()
    own = emb is None
    emb = emb or ctx.embedder()
    try:
        P = cached_encode(emb, [ctx.index_text(p) for p in passages], PASSAGE_MAXLEN, "passages", ctx.work / "emb")
        Qhi = cached_encode(emb, [s["question_hi"] for s in sets], QUERY_MAXLEN, "q_hi_" + HAIKU, ctx.work / "emb")
        Qen = cached_encode(emb, [s["question_en"] for s in sets], QUERY_MAXLEN, "q_en_" + HAIKU, ctx.work / "emb")
        set_vecs = {s["set_id"]: (Qhi[i], Qen[i]) for i, s in enumerate(sets)}
        rows, qs, runs = stage.search_test(emb, passages, sets, questions, set_vecs, P)
    finally:
        if own:
            emb.close()
    (ctx.work / "search.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (ctx.work / "search_runs.json").write_text(json.dumps({"qids": [q["qid"] for q in qs], "runs": runs}),
                                              encoding="utf-8")
    print("search test on %d questions with the gold chunk in the sub-corpus" % len(qs))
    return rows


def do_review(ctx):
    passages = ctx.passages()
    sh = read_jsonl(ctx.sets_path(HAIKU))
    ss = read_jsonl(ctx.sets_path(SONNET))
    ids = set(ctx.review_ids(passages))
    nh, ns = stage.review_files(passages, [s for s in sh if s["chunk_id"] in ids], ss,
                                ctx.final / "sets-for-review.csv", ctx.final / "sets-for-review-answer-key.csv")
    print("review file: %d haiku + %d sonnet sets" % (nh, ns))


RESULT_COLS = ["session", "stage", "run_id", "setup", "corpus_size", "question_group", "level", "recall_5",
               "recall_10", "recall_20", "mrr_10", "ndcg_10", "p50_ms", "p95_ms", "paid_usd_per_1000", "machine",
               "notes"]


def do_results(ctx):
    if not ctx.dry:
        return do_results_shared(ctx)
    rows = json.loads((ctx.work / "search.json").read_text(encoding="utf-8"))
    n_pass = len(ctx.passages())
    mach = machine_string()
    gen_cost = None
    try:
        g = json.loads(ctx.gen_path(HAIKU).read_text(encoding="utf-8"))
        from dutil import usd
        gen_cost = 1000 * usd(HAIKU, sum(r["in_tokens"] for r in g.values()),
                              sum(r["out_tokens"] for r in g.values()), True) / max(1, len(g))
    except Exception:
        pass
    setups = []
    for r in rows:
        if r["setup"] not in setups:
            setups.append(r["setup"])
    with open(ctx.results / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLS)
        w.writeheader()
        for r in rows:
            note = "%d questions; %d index vectors; %d sets" % (r["n_questions"], r["n_vectors"], r["n_sets"])
            if gen_cost is not None and r["n_sets"]:
                note += "; one-off set generation USD %.3f per 1000 passages (haiku batch)" % gen_cost
            w.writerow({
                "session": "D", "stage": "D5", "run_id": "D5-%02d" % (setups.index(r["setup"]) + 1),
                "setup": "bge-m3 dense; " + r["setup"], "corpus_size": n_pass,
                "question_group": r["question_group"], "level": r["level"],
                **{k: ("%.4f" % r[k]) for k in ("recall_5", "recall_10", "recall_20", "mrr_10", "ndcg_10")},
                "p50_ms": "%.1f" % r["p50_ms"], "p95_ms": "%.1f" % r["p95_ms"], "paid_usd_per_1000": "0",
                "machine": mach, "notes": note})
    print("wrote", ctx.results / "results.csv")


def gen_cost_per_1000(ctx):
    from dutil import usd
    g = json.loads(ctx.gen_path(HAIKU).read_text(encoding="utf-8"))
    return 1000 * usd(HAIKU, sum(r["in_tokens"] for r in g.values()),
                      sum(r["out_tokens"] for r in g.values()), True) / max(1, len(g))


def do_results_shared(ctx):
    """results/D/results.csv with the shared scorer (src.common.score), plus the same- vs
    cross-language split for findings."""
    from src.common.score import result_rows, score_run, write_results
    sr = json.loads((ctx.work / "search_runs.json").read_text(encoding="utf-8"))
    ids = {p["chunk_id"] for p in ctx.passages()}
    chunks = data.chunks_df()
    chunks = chunks[chunks.chunk_id.isin(ids)]
    q = data.questions_df()
    q = q[q.qid.isin(set(sr["qids"]))]
    out = ctx.results / "results.csv"
    if out.exists():
        out.unlink()
    gc = gen_cost_per_1000(ctx)
    split = []
    for i, r in enumerate(sr["runs"], 1):
        note = "sub-corpus %d chunks; %d index vectors; %d sets" % (len(ids), r["n_vectors"], r["n_sets"])
        if r["n_sets"]:
            note += "; one-off set generation USD %.2f per 1000 passages (haiku batch)" % gc
        rows = result_rows(r["ranked"], chunks, session="D", stage="D5", run_id="D5-%02d" % i,
                           setup="bge-m3 dense; " + r["setup"], corpus_size="1.5k",
                           latencies_ms=r["latencies_ms"], paid_usd_per_1000=0.0, notes=note, questions=q)
        write_results(rows, out)
        lm = score_run(r["ranked"], chunks, q, by="lang_match")
        lm.insert(0, "setup", r["setup"])
        split.append(lm)
    import pandas as pd
    pd.concat(split).to_csv(ctx.work / "search_lang_match.csv", index=False)
    print("wrote", out)


def do_dryrun(base):
    ctx = Ctx(dry=True, base=base)
    emb = FakeEmbedder()
    do_generate(ctx, HAIKU)
    do_generate(ctx, SONNET)
    do_filter(ctx, HAIKU, emb)
    do_filter(ctx, SONNET, emb)
    do_report(ctx)
    do_scale(ctx)
    do_search(ctx, emb)
    do_review(ctx)
    do_results(ctx)


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--base", default="")
    a = ap.parse_args()
    if a.cmd == "dryrun":
        do_dryrun(a.base or str(Path.home() / "c2_D_dryrun"))
        return
    ctx = Ctx()
    m = MODELS[a.model]
    if a.cmd == "select":
        data.select_subcorpus(ctx.work)
    elif a.cmd == "estimate":
        ps = ctx.passages()
        if m == SONNET:
            ids = set(ctx.review_ids(ps))
            ps = [p for p in ps if p["chunk_id"] in ids]
        e = gen.estimate_cost(m, ps, out_per_passage=a.n if a.n > 50 else 1500, batch=True)
        print(json.dumps(e, indent=1))
    elif a.cmd == "pilot":
        ps = ctx.passages()
        rnd = random.Random(SEED)
        pick = rnd.sample(sorted(ps, key=lambda p: p["chunk_id"]), a.n)
        res = gen.run_sync(m, pick, "D1 pilot %s" % m)
        (ctx.work / ("pilot_%s.json" % m.split("-")[1])).write_text(json.dumps(res, ensure_ascii=False),
                                                                     encoding="utf-8")
    elif a.cmd == "generate":
        do_generate(ctx, m, use_batch=not a.sync)
    elif a.cmd == "filter":
        do_filter(ctx, m)
    elif a.cmd == "report":
        do_report(ctx)
    elif a.cmd == "scale":
        do_scale(ctx)
    elif a.cmd == "search":
        do_search(ctx)
    elif a.cmd == "review":
        do_review(ctx)
    elif a.cmd == "results":
        do_results(ctx)
    else:
        sys.exit("unknown command " + a.cmd)


if __name__ == "__main__":
    main()
