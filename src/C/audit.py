"""Stage A data audit (session C). No paid calls.

    python -m src.C.audit                 # tables must exist: python -m src.common.load
    python -m src.C.audit --tokens        # also count tokens on 200 sampled records (Anthropic
                                          # token counting, free) and bge-m3 tokens on everything

Writes ../local/final/data-audit.md and ../local/work/audit/*.parquet|json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import time
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import load as L

FINAL = L.LOCAL / "final"
AUD = L.WORK / "audit"
SEED = 7

# ---------------------------------------------------------------- column roles and glosses
GLOSS = [
    (r"^Mantra Ref ID$", "verse reference"),
    (r"^(Mandal|Sukta|Mantra No|Kand|Paryayah|Adhyay|Mantra Sankhya)$", "verse number part"),
    (r"^(Devata|Rishi|Chhanda?|Swara)$|ऋषि \| देवता", "verse metadata (deity, seer, metre, note)"),
    (r"^स्वर रहित मन्त्र$", "verse text, no accent marks"),
    (r"^Mantra with Swara$", "verse text with accent marks"),
    (r"मन्त्र \(बांग्ला\)|^मंत्र$|^মন্ত্র$", "verse text (other script or copy)"),
    (r"^स्वर सहित पद पाठ$", "pada (word-by-word) text with accents"),
    (r"^स्वर रहित पद पाठ$", "pada (word-by-word) text, no accents"),
    (r"अन्यत्र|অন্যত্র", "where else the verse is explained"),
    (r"भाष्यसार|ভাষ্যসার", "summary of the commentary"),
    (r"संस्कृत भाग", "Sanskrit part of the commentary"),
    (r"व्यक्तव्य", "remarks / statement"),
    (r"विषय|Subject|বিষয়|Mazmoon|Vishay|^सूक्त$", "subject / topic line"),
    (r"न्वय|Anvaya", "prose word order (anvaya), may carry meanings"),
    (r"पदार्थ|Padartha|Padarth|পদার্থ|Word Meaning|शब्दार्थ|Lafzi|Literal meaning|Etymology", "word-by-word meaning"),
    (r"भावार्थ|Bhavarth|ভাবার্থ|Tashree|Purport|gist|Essence", "purport / gist"),
    (r"टिप्पणी|टिपण्णी|Tippani|টিপ্পনী|Note|Footnote|FootNotes|Comment|Tika", "notes / footnotes"),
    (r"Translation|Meaning|भाषार्थ|मन्त्रार्थ|अनुवाद|বঙ্গানুবাদ|সরলার্থ|सूक्तार्थ|TRAN|भाषा भाष्य|Bhasha", "translation of the verse"),
    (r"विशेष|सूचना|Khaas|Vishesh", "special remarks"),
    (r"व्याख्या|व्याखान|Vyakhya|Reflection|Commentary|Explanation|ভাষ্য$|Vinay", "explanation / discourse"),
    (r"Bhajan", "devotional song on the verse"),
    (r"प्रमाणार्थ", "cited authorities"),
]


def gloss(col: str) -> str:
    col = unicodedata.normalize("NFC", col)
    for pat, g in GLOSS:
        if re.search(unicodedata.normalize("NFC", pat), col, re.I):
            return g
    return "?"


def role(col: str) -> str:
    base = col.split("__")[0]
    if base in L.META_COLS:
        return "meta"
    if base in L.VERSE_TEXT_COLS:
        return "verse"
    if base in L.PADA_COLS:
        return "pada"
    return "text"


# ---------------------------------------------------------------- scripts
SCRIPTS = [("deva", 0x0900, 0x097F), ("deva", 0xA8E0, 0xA8FF), ("deva", 0x1CD0, 0x1CFF),
           ("beng", 0x0980, 0x09FF), ("guj", 0x0A80, 0x0AFF), ("tamil", 0x0B80, 0x0BFF),
           ("arabic", 0x0600, 0x06FF), ("latin", 0x0041, 0x005A), ("latin", 0x0061, 0x007A),
           ("latin", 0x00C0, 0x024F), ("latin", 0x1E00, 0x1EFF)]


def script_counts(s: str, acc: collections.Counter):
    for ch in s:
        o = ord(ch)
        if o < 0x41 or ch.isspace():
            continue
        for name, lo, hi in SCRIPTS:
            if lo <= o <= hi:
                acc[name] += 1
                break


def script_mix(acc: collections.Counter, top=3) -> str:
    tot = sum(acc.values())
    if not tot:
        return "-"
    return " ".join(f"{k}:{100*v/tot:.0f}%" for k, v in acc.most_common(top) if 100 * v / tot >= 1)


# ---------------------------------------------------------------- damage checks
ZW = {"​": "ZWSP", "‌": "ZWNJ", "‍": "ZWJ", "﻿": "BOM", "⁠": "WJ", "­": "SHY"}
HTML_TAG = re.compile(r"</?(?:p|br|div|span|b|i|u|strong|em|font|sup|sub|a|li|ul|ol|table|tr|td|th|h\d|img)\b[^>]*>", re.I)
HTML_ENT = re.compile(r"&(?:nbsp|amp|lt|gt|quot|apos|#\d+|#x[0-9a-f]+);", re.I)
XL_ERR = {"#N/A", "#VALUE!", "#REF!", "#DIV/0!", "#NAME?", "#NUM!", "#NULL!", "#SPILL!", "#CALC!"}
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
LITERAL_NULL = {"undefined", "nan", "none", "null", "n/a", "-"}


def cell_checks(s: str, c: collections.Counter, det: collections.Counter | None = None, where: str = ""):
    """Count damage in one cell. det collects (check, where, detail) examples for the report."""
    if not s:
        return
    n = len(s)
    if n == 32767:
        c["cut_at_32767"] += 1
    elif n >= 32000:
        c["len_ge_32000"] += 1
    if s.startswith("="):
        c["starts_with_equals"] += 1
    if s.strip() in XL_ERR:
        c["excel_error_value"] += 1
    if not unicodedata.is_normalized("NFC", s):
        c["not_NFC"] += 1
        if det is not None:
            for ch in set(s) - set(unicodedata.normalize("NFC", s)):
                det[("not_NFC", "", f"U+{ord(ch):04X} {unicodedata.name(ch, '?')}")] += 1
    for ch, name in ZW.items():
        if ch in s:
            c[f"zw_{name}"] += 1
    m = HTML_TAG.search(s)
    if m:
        c["html_tag"] += 1
        if det is not None:
            det[("html_tag", where, m.group(0)[:12] + (" (whole cell)" if s.strip() == m.group(0) else ""))] += 1
    if HTML_ENT.search(s):
        c["html_entity"] += 1
    if "�" in s:
        c["replacement_char"] += 1
    m = CTRL.search(s)
    if m:
        c["control_char"] += 1
        if det is not None:
            det[("control_char", where, f"U+{ord(m.group(0)):04X}")] += 1
    if "\xa0" in s:
        c["nbsp"] += 1
    if "\r" in s:
        c["carriage_return"] += 1
    if s.strip().lower() in LITERAL_NULL:
        c["literal_null_text"] += 1
        if det is not None:
            det[("literal_null_text", where, repr(s.strip()))] += 1
    if n >= 32000 and det is not None:
        det[("long_cell", where, f"{n} chars")] += 1
    if s != s.strip():
        c["edge_whitespace"] += 1


def xml_scan(path: Path) -> dict:
    """Formulas and error cells straight from the sheet XML."""
    out = {"formulas": 0, "error_cells": 0, "date_styles": 0}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"):
                x = z.read(n).decode("utf-8", "replace")
                out["formulas"] += len(re.findall(r"<f[ >/]", x))
                out["error_cells"] += len(re.findall(r'\bt="e"', x))
    return out


# ---------------------------------------------------------------- record text and sizes
SENT_SPLIT = re.compile(r"[।॥?!]+|\.(?=\s|$)")


def n_sentences(s: str) -> int:
    return sum(1 for p in SENT_SPLIT.split(s) if p.strip() and re.search(r"\w", p))


def record_text(row: dict, text_cols: list[str]) -> str:
    parts = [row[c].strip() for c in text_cols if row.get(c) and row[c].strip()]
    return "\n".join(parts)


def dist(a) -> dict:
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return {}
    q = np.percentile(a, [10, 50, 90, 99])
    return dict(n=int(a.size), mean=float(a.mean()), p10=float(q[0]), p50=float(q[1]),
                p90=float(q[2]), p99=float(q[3]), max=float(a.max()))


def fmt_d(d: dict) -> str:
    if not d:
        return "-"
    return f"{d['mean']:.0f} | {d['p10']:.0f} / {d['p50']:.0f} / {d['p90']:.0f} / {d['max']:.0f}"


# ---------------------------------------------------------------- main audit
def run(tokens: bool):
    AUD.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    idx = L.load_index()
    files = {f["table_id"]: f for f in L.list_files()}

    # samhita verse sets
    samhita_refs, samhita_info = {}, {}
    for coll in L.SAMHITA_FILES:
        df = L.load_table(f"samhita.{coll}")
        refs = df["Mantra Ref ID"].map(L.norm_ref)
        samhita_refs[coll] = set(refs[refs != ""])
        samhita_info[coll] = df

    col_rows, table_rows, rec_rows, damage_rows, link_rows = [], [], [], [], []
    ref_damage = []
    raw_type_notes = []
    text_hash_tables = collections.defaultdict(set)
    details: collections.Counter = collections.Counter()

    for _, t in idx.iterrows():
        tid = t.table_id
        f = files[tid]
        df = L.load_table(tid)
        cols = [c for c in df.columns if not c.startswith("_")]
        # raw cell types straight from Excel
        raw = L.read_rows(f["path"])
        hdr = L.dedupe_headers(raw[0])
        for j, h in enumerate(hdr):
            tc = collections.Counter(type(r[j]).__name__ if j < len(r) else "NoneType" for r in raw[1:])
            odd = {k: v for k, v in tc.items() if k not in ("str", "NoneType")}
            if odd and not (h in ("Mandal", "Sukta", "Mantra No", "Kand", "Adhyay", "Mantra Sankhya")):
                ex = [repr(r[j]) for r in raw[1:] if j < len(r) and type(r[j]).__name__ in odd][:4]
                raw_type_notes.append(dict(table_id=tid, column=h, types=dict(tc), examples=ex))
        xs = xml_scan(f["path"])

        text_cols = [c for c in cols if role(c) == "text"]
        dmg = collections.Counter()
        for c in cols:
            s = df[c]
            acc = collections.Counter()
            lens = s.str.len()
            nonempty = s.str.strip() != ""
            for v in s[nonempty].head(4000):
                script_counts(v, acc)
            for v in s:
                cell_checks(v, dmg, details, f"{tid[:45]} / {c}")
            sample = next((v for v in s if v.strip()), "")
            col_rows.append(dict(table_id=tid, column=c, role=role(c), gloss=gloss(c),
                                 nonempty_pct=100 * nonempty.mean() if len(s) else 0,
                                 mean_chars=float(lens[nonempty].mean()) if nonempty.any() else 0,
                                 max_chars=int(lens.max()) if len(s) else 0,
                                 scripts=script_mix(acc), dup_pct=100 * s[nonempty].duplicated().mean() if nonempty.any() else 0,
                                 sample=sample.replace("\n", " ")[:50]))

        # references
        nref = df["Mantra Ref ID"].map(L.norm_ref)
        bad = df.loc[nref == "", "Mantra Ref ID"]
        for v in bad.head(5):
            ref_damage.append(dict(table_id=tid, kind="unparseable", value=v[:40]))
        if len(bad):
            dmg["ref_unparseable"] += len(bad)
        rawref = [r[0] for r in raw[1:] if r and r[0] is not None]
        frac = [x for x in rawref if isinstance(x, float) and not x.is_integer()]
        if frac:
            dmg["ref_float_with_fraction"] += len(frac)
            ref_damage.append(dict(table_id=tid, kind="float_with_fraction", value=str(frac[:5])))
        n_float = sum(isinstance(x, float) for x in rawref)
        if n_float and f["collection"] not in ("sama",):
            dmg["ref_stored_as_number"] += n_float
        dts = [x for x in rawref if type(x).__name__ in ("datetime", "date", "time", "timedelta")]
        if dts:
            dmg["ref_stored_as_date"] += len(dts)
            ref_damage.append(dict(table_id=tid, kind="date", value=str(dts[:5])))

        # duplicates
        key_cols = cols
        dup_rows = int(df.duplicated(subset=key_cols).sum())
        dup_refs = int(nref[nref != ""].duplicated().sum())

        # links
        coll = f["collection"]
        if f["kind"] == "bhashya":
            assigned = []
            for r in nref:
                if not r:
                    assigned.append(("", "unparseable"))
                    continue
                cands = ref_collection(r) if coll == "multi" else [coll]
                hits = [c for c in cands if r in samhita_refs[c]]
                if len(hits) == 1:
                    assigned.append((hits[0], "linked"))
                elif len(hits) > 1:
                    assigned.append(("|".join(hits), "ambiguous"))
                else:
                    assigned.append(("|".join(cands) or "?", "broken"))
            st = collections.Counter(a[1] for a in assigned)
            broken_ex = [r for r, a in zip(nref, assigned) if a[1] == "broken"][:6]
            link_rows.append(dict(table_id=tid, collection=coll, records=len(df), linked=st["linked"],
                                  ambiguous=st["ambiguous"], broken=st["broken"], unparseable=st["unparseable"],
                                  broken_examples=" ".join(broken_ex)))
            vcoll = [a[0] for a in assigned]
            # record-level text stats
            for i, row in enumerate(df.to_dict("records")):
                txt = record_text(row, text_cols)
                h = hashlib.sha1(txt.encode("utf-8")).hexdigest() if txt else ""
                if h:
                    text_hash_tables[h].add(tid)
                rec_rows.append(dict(table_id=tid, author_id=f["author_id"], lang=f["lang"],
                                     collection=coll, verse_coll=vcoll[i], link=assigned[i][1],
                                     ref=nref.iloc[i], row=int(row["_row"]),
                                     chars=len(txt), words=len(txt.split()), sents=n_sentences(txt),
                                     h=h))
        else:
            # samhita: reference must agree with the number columns
            mism = 0
            parts = {"rig": ["Mandal", "Sukta", "Mantra No"], "yajur": ["Adhyay", "Mantra No"],
                     "atharva": ["Kand", "Sukta", "Mantra No"], "sama": ["Mantra Sankhya"]}[coll]
            for row, r in zip(df.to_dict("records"), nref):
                exp = "/".join(str(row[p]) for p in parts)
                got = r
                if coll == "atharva" and r.count("/") == 3:
                    a = r.split("/")
                    got = "/".join([a[0], a[1], a[3]])
                if exp != got:
                    mism += 1
                    if mism <= 5:
                        ref_damage.append(dict(table_id=tid, kind="ref_vs_number_columns",
                                               value=f"ref={row['Mantra Ref ID']} cols={exp}"))
            if mism:
                dmg["ref_vs_number_columns_mismatch"] += mism

        empty_text = int(sum(1 for row in df[text_cols].itertuples(index=False)
                             if not any(str(v).strip() for v in row))) if text_cols else len(df)
        table_rows.append(dict(table_id=tid, kind=f["kind"], collection=coll, lang=f["lang"],
                               author_id=f["author_id"], rows=len(df), cols=len(cols),
                               text_cols=len(text_cols), dup_rows=dup_rows, dup_refs=dup_refs,
                               empty_text_rows=empty_text, copies=len(f["copies"]),
                               formulas=xs["formulas"], error_cells=xs["error_cells"]))
        for k, v in dmg.items():
            damage_rows.append(dict(table_id=tid, check=k, cells=v))
        print(f"audited {tid}")

    cols_df = pd.DataFrame(col_rows)
    tab_df = pd.DataFrame(table_rows)
    rec_df = pd.DataFrame(rec_rows)
    dmg_df = pd.DataFrame(damage_rows)
    link_df = pd.DataFrame(link_rows)
    rec_df["cross_work_dup"] = rec_df["h"].map(lambda h: len(text_hash_tables.get(h, ())) > 1 if h else False)
    for name, d in [("columns", cols_df), ("tables", tab_df), ("records", rec_df), ("damage", dmg_df), ("links", link_df)]:
        d.to_parquet(AUD / f"{name}.parquet", index=False)
    (AUD / "raw_type_notes.json").write_text(json.dumps(raw_type_notes, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.DataFrame([dict(check=k[0], where=k[1], detail=k[2], cells=v) for k, v in details.items()]).to_parquet(
        AUD / "damage_details.parquet", index=False)
    (AUD / "ref_damage.json").write_text(json.dumps(ref_damage, ensure_ascii=False, indent=1), encoding="utf-8")

    tok = token_estimates(rec_df) if tokens else json.loads((AUD / "tokens.json").read_text()) if (AUD / "tokens.json").exists() else None
    write_report(idx, cols_df, tab_df, rec_df, dmg_df, link_df, raw_type_notes, ref_damage, samhita_info, tok)


def ref_collection(r):
    return L.ref_collection(r)


# ---------------------------------------------------------------- tokens
def token_estimates(rec_df: pd.DataFrame) -> dict:
    """200 sampled records counted with Anthropic token counting, scaled by language.
    bge-m3 tokens counted exactly for every record."""
    import os
    from dotenv import dotenv_values
    import anthropic
    from transformers import AutoTokenizer

    tables = {}

    def text_of(tid, row):
        if tid not in tables:
            df = L.load_table(tid)
            cols = [c for c in df.columns if not c.startswith("_") and role(c) == "text"]
            tables[tid] = (df.set_index("_row"), cols)
        df, cols = tables[tid]
        return record_text(df.loc[row].to_dict(), cols)

    nonempty = rec_df[rec_df.chars > 0].copy()
    nonempty["lgrp"] = nonempty.lang.where(nonempty.lang.isin(["hi", "sa", "en"]), "other")
    rng = random.Random(SEED)
    sample = []
    for g, n in [("hi", 60), ("sa", 50), ("en", 50), ("other", 40)]:
        ids = sorted(nonempty.index[nonempty.lgrp == g].tolist())
        sample += [(g, i) for i in rng.sample(ids, n)]

    key = dotenv_values(L.LOCAL / ".env").get("VEDIC_ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=key)
    model = "claude-haiku-5-5"
    base = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": "x"}]).input_tokens - 1
    tokz = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    rows = []
    for g, i in sample:
        r = nonempty.loc[i]
        txt = text_of(r.table_id, r.row)
        for attempt in range(6):
            try:
                n = client.messages.count_tokens(model=model, messages=[{"role": "user", "content": txt}]).input_tokens - base
                break
            except anthropic.RateLimitError:
                time.sleep(5 + 5 * attempt)
        rows.append(dict(lgrp=g, chars=len(txt), claude=n, bge=len(tokz(txt, add_special_tokens=False)["input_ids"])))
        time.sleep(0.25)
    s = pd.DataFrame(rows)
    s.to_parquet(AUD / "token_sample.parquet", index=False)

    # bge-m3 tokens for every record (exact)
    bge_all = {}
    for tid, grp in nonempty.groupby("table_id"):
        df = L.load_table(tid)
        cols = [c for c in df.columns if not c.startswith("_") and role(c) == "text"]
        df = df.set_index("_row")
        txts = [record_text(df.loc[r].to_dict(), cols) for r in grp.row]
        n = 0
        for k in range(0, len(txts), 256):
            n += sum(len(x) for x in tokz(txts[k:k + 256], add_special_tokens=False)["input_ids"])
        bge_all[tid] = n
    nonempty["bge_table"] = nonempty.table_id.map(bge_all)

    out = {"model": model, "sample_n": len(s), "per_group": {}, "total": {}}
    totals = nonempty.groupby("lgrp").chars.sum()
    for g, d in s.groupby("lgrp"):
        ratio = d.claude.sum() / d.chars.sum()
        bratio = d.bge.sum() / d.chars.sum()
        out["per_group"][g] = dict(n=len(d), claude_per_char=ratio, bge_per_char=bratio,
                                   corpus_chars=int(totals[g]),
                                   est_claude_tokens=int(ratio * totals[g]))
    out["total"]["est_claude_tokens"] = sum(v["est_claude_tokens"] for v in out["per_group"].values())
    out["total"]["bge_tokens_exact"] = int(sum(bge_all.values()))
    out["bge_per_author"] = {}
    agg = collections.defaultdict(int)
    for tid, n in bge_all.items():
        agg[nonempty.loc[nonempty.table_id == tid, "author_id"].iloc[0]] += n
    out["bge_per_author"] = dict(sorted(agg.items(), key=lambda kv: -kv[1]))
    out["bge_per_table"] = bge_all
    (AUD / "tokens.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ---------------------------------------------------------------- report
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(
            ((f"{v:.3g}" if abs(v) < 10 else f"{v:.1f}") if isinstance(v, float) else str(v)).replace("|", "/").replace("\n", " ") for v in r) + " |")
    return "\n".join(out)


def write_report(idx, cols_df, tab_df, rec_df, dmg_df, link_df, raw_type_notes, ref_damage, samhita_info, tok):
    W = []
    bh = tab_df[tab_df.kind == "bhashya"]
    n_files = sum(t["copies"] for t in tab_df.to_dict("records"))
    W.append("# Data audit (Stage A)\n")
    W.append(f"Generated by `python -m src.C.audit` on {time.strftime('%Y-%m-%d %H:%M')}. Local only, never in Git.\n")
    W.append("## 1. Summary\n")
    W.append(f"- {n_files} Excel files, one sheet each (`Worksheet`), in `samhitas/` (4 verse tables) and "
             f"`bhashyas/<veda>/` ({n_files - 4} commentary files).")
    W.append(f"- {int((tab_df.copies > 1).sum())} commentary works are stored as identical copies in 2 to 4 Veda folders. "
             f"They are loaded once with collection `multi`. That leaves {len(bh)} distinct commentary tables.")
    W.append(f"- Verses: " + ", ".join(f"{c} {len(samhita_info[c])}" for c in samhita_info) +
             f" = {sum(len(v) for v in samhita_info.values())}.")
    W.append(f"- Commentary records (one row = one verse in one work): {len(rec_df)}, "
             f"of which {int((rec_df.chars > 0).sum())} have text. Total text {rec_df.chars.sum()/1e6:.1f} M characters.")
    by_lang = rec_df.groupby("lang").agg(records=("chars", "size"), Mchars=("chars", lambda x: x.sum() / 1e6),
                                         works=("table_id", "nunique")).sort_values("Mchars", ascending=False)
    W.append("- By language (from the file name; checked against the script of the text):\n")
    W.append(md_table(by_lang.reset_index()))
    W.append("")
    if tok:
        W.append(f"- Estimated tokens for all commentary text: about {tok['total']['est_claude_tokens']/1e6:.1f} M "
                 f"Claude tokens; {tok['total']['bge_tokens_exact']/1e6:.1f} M bge-m3 tokens (exact count).")
    W.append("")

    W.append("## 2. Tables\n")
    t = tab_df.copy()
    t["table_id"] = t.table_id.str.slice(0, 70)
    W.append(md_table(t[["table_id", "kind", "collection", "lang", "author_id", "rows", "cols", "text_cols",
                         "empty_text_rows", "dup_rows", "dup_refs", "copies"]]))
    W.append("")

    W.append("## 3. Columns: what each holds\n")
    W.append("Distinct column headers across all tables, with what they appear to hold, how many tables use them, "
             "how often they are filled, and the scripts found. Per-table detail is in Appendix A.\n")
    g = cols_df.groupby("column").agg(tables=("table_id", "nunique"), role=("role", "first"), gloss=("gloss", "first"),
                                      nonempty_pct=("nonempty_pct", "mean"), mean_chars=("mean_chars", "mean"),
                                      scripts=("scripts", lambda x: collections.Counter(x).most_common(1)[0][0]))
    g = g.sort_values(["role", "tables"], ascending=[True, False]).reset_index()
    W.append(md_table(g))
    W.append("")

    W.append("## 4. How the tables link\n")
    W.append("- There is no separate author or verse key table. The link is the column `Mantra Ref ID` in every file.")
    W.append("- Verse key = Veda + reference. Reference shapes: Rigveda `mandala/sukta/mantra`; Atharvaveda "
             "`kanda/sukta/mantra`, or `kanda/sukta/paryaya/mantra` for the 343 paryaya verses; Yajurveda "
             "`adhyaya/mantra`; Samaveda a running number 1 to 1875, stored by Excel as a number (`1.0`).")
    W.append("- The Veda of a commentary comes from its folder. For the `multi` works it is inferred per row from the "
             "reference shape: 1 part Sama, 2 parts Yajur, 4 parts Atharva, 3 parts Rig or Atharva (ambiguous unless "
             "the reference exists in only one of the two samhitas).")
    W.append("- Author: `author_id` is the file name as it is (one file = one work); no names are merged. Possible "
             "matches (the same person, spelling variants) are listed in ../local/final/open-questions.md for the society.\n")
    lk = link_df.copy()
    tot = lk[["records", "linked", "ambiguous", "broken", "unparseable"]].sum()
    W.append(f"Totals: {int(tot.records)} records, {int(tot.linked)} linked to a verse, {int(tot.ambiguous)} ambiguous "
             f"(multi works, Rig or Atharva), {int(tot.broken)} broken (reference not in the samhita), "
             f"{int(tot.unparseable)} unparseable references.\n")
    lk_bad = lk[(lk.broken + lk.unparseable + lk.ambiguous) > 0].copy()
    lk_bad["table_id"] = lk_bad.table_id.str.slice(0, 70)
    W.append("Tables with broken, ambiguous or unparseable links:\n")
    W.append(md_table(lk_bad[["table_id", "collection", "records", "linked", "ambiguous", "broken", "unparseable", "broken_examples"]]))
    W.append("")
    # verses -> number of commentaries
    W.append("Commentaries per verse (linked records only):\n")
    lr = rec_df[rec_df.link == "linked"]
    rows = []
    for c in ["rig", "sama", "yajur", "atharva"]:
        sub = lr[lr.verse_coll == c]
        per = sub.groupby("ref").table_id.nunique()
        allv = len(samhita_info[c])
        zero = allv - len(per)
        rows.append(dict(collection=c, verses=allv, verses_with_commentary=len(per), verses_without=zero,
                         works=sub.table_id.nunique(), records=len(sub),
                         mean_works_per_verse=float(per.mean()) if len(per) else 0,
                         median=float(per.median()) if len(per) else 0, max=int(per.max()) if len(per) else 0))
    W.append(md_table(pd.DataFrame(rows)))
    W.append("")
    W.append("Counts per collection (by folder; `multi` = anthologies stored in several folders):\n")
    pc = rec_df.groupby("collection").agg(works=("table_id", "nunique"), authors=("author_id", "nunique"),
                                         records=("chars", "size"), Mchars=("chars", lambda x: x.sum() / 1e6)).reset_index()
    W.append(md_table(pc))
    W.append("")
    W.append("Counts per work (author_id = the file name; one file = one work):\n")
    pa = rec_df.groupby("author_id").agg(works=("table_id", "nunique"),
                                        collections=("collection", lambda x: ",".join(sorted(set(x)))),
                                        records=("chars", "size"), with_text=("chars", lambda x: int((x > 0).sum())),
                                        Mchars=("chars", lambda x: x.sum() / 1e6)).sort_values("Mchars", ascending=False).reset_index()
    W.append(md_table(pa))
    W.append("")

    W.append("## 5. Excel damage and text hygiene\n")
    W.append("Counts are cells. `not_NFC` cells change under NFC normalization. ZWJ/ZWNJ have legitimate uses in "
             "Indic scripts; ZWSP, BOM, WJ and soft hyphen do not.\n")
    if len(dmg_df):
        pv = dmg_df.pivot_table(index="check", values="cells", aggfunc="sum").sort_values("cells", ascending=False)
        pv["tables"] = dmg_df.groupby("check").table_id.nunique()
        W.append(md_table(pv.reset_index()))
    W.append("")
    dd = pd.read_parquet(AUD / "damage_details.parquet")
    W.append("Details of the flagged cells (where, what, how many):\n")
    nf = dd[dd.check == "not_NFC"].groupby("detail").cells.sum().sort_values(ascending=False).head(8)
    W.append("- Characters that NFC changes (precomposed nukta letters and decomposed Latin diacritics): " +
             "; ".join(f"{k} ({v})" for k, v in nf.items()) + ".")
    other = dd[dd.check != "not_NFC"].sort_values("cells", ascending=False).head(25)
    W.append("")
    W.append(md_table(other[["check", "where", "detail", "cells"]]))
    W.append("")
    W.append(f"- Formulas found in sheet XML: {int(tab_df.formulas.sum())}. Error cells (`t=\"e\"`): {int(tab_df.error_cells.sum())}.")
    W.append(f"- Cells cut at 32,767 characters: {int(dmg_df.loc[dmg_df.check == 'cut_at_32767', 'cells'].sum()) if len(dmg_df) else 0}.")
    W.append("- Reference problems (first examples per table):\n")
    if ref_damage:
        rd = pd.DataFrame(ref_damage)
        rd["table_id"] = rd.table_id.str.slice(0, 60)
        W.append(md_table(rd))
    W.append("")
    W.append("- Cells whose raw Excel type is a number or date where text is expected:\n")
    if raw_type_notes:
        rt = pd.DataFrame(raw_type_notes)
        rt["table_id"] = rt.table_id.str.slice(0, 60)
        rt["types"] = rt.types.map(lambda d: ", ".join(f"{k}:{v}" for k, v in d.items()))
        rt["examples"] = rt.examples.map(lambda x: " ".join(x))
        W.append(md_table(rt))
    W.append("")
    W.append("Per-table damage counts:\n")
    if len(dmg_df):
        pt = dmg_df.pivot_table(index="table_id", columns="check", values="cells", aggfunc="sum", fill_value=0)
        pt.index = [i[:55] for i in pt.index]
        W.append(md_table(pt.reset_index().rename(columns={"index": "table_id"})))
    W.append("")
    W.append(f"- Records whose full text also appears in another work: {int(rec_df.cross_work_dup.sum())}.")
    W.append("")

    W.append("## 6. Text length per author\n")
    W.append("Per record (all commentary columns joined). Each cell: mean | p10 / median / p90 / max. "
             "Sentences split on danda (। ॥), ? ! and full stop.\n")
    rows = []
    for a, d in rec_df[rec_df.chars > 0].groupby("author_id"):
        rows.append(dict(author_id=a, records=len(d), chars=fmt_d(dist(d.chars)), words=fmt_d(dist(d.words)),
                         sentences=fmt_d(dist(d.sents))))
    W.append(md_table(pd.DataFrame(rows).sort_values("records", ascending=False)))
    W.append("")
    W.append("Overall characters per record by language:\n")
    rows = []
    for lg, d in rec_df[rec_df.chars > 0].groupby("lang"):
        dd = dist(d.chars)
        rows.append(dict(lang=lg, records=len(d), mean=dd["mean"], p10=dd["p10"], p50=dd["p50"], p90=dd["p90"],
                         p99=dd["p99"], max=dd["max"]))
    W.append(md_table(pd.DataFrame(rows)))
    W.append("")

    W.append("## 7. Tokens\n")
    if tok:
        W.append(f"Anthropic token counting (`{tok['model']}`) on {tok['sample_n']} records sampled with seed 7 "
                 "(60 Hindi, 50 Sanskrit, 50 English, 40 other languages), scaled by characters per language group. "
                 "bge-m3 tokens were counted on every record.\n")
        rows = [dict(group=g, sample=v["n"], claude_tokens_per_char=v["claude_per_char"],
                     bge_tokens_per_char=v["bge_per_char"], corpus_Mchars=v["corpus_chars"] / 1e6,
                     est_claude_Mtokens=v["est_claude_tokens"] / 1e6) for g, v in tok["per_group"].items()]
        W.append(md_table(pd.DataFrame(rows)))
        W.append("")
        W.append(f"Estimated total: {tok['total']['est_claude_tokens']/1e6:.1f} M Claude tokens. "
                 f"bge-m3 exact total: {tok['total']['bge_tokens_exact']/1e6:.1f} M tokens, i.e. about "
                 f"{tok['total']['bge_tokens_exact']/300/1e3:.0f} k chunks of 300 tokens.\n")
        W.append("bge-m3 tokens per work (top 25):\n")
        per_work = collections.Counter()
        for tid, n in tok["bge_per_table"].items():
            per_work[L.load_index().set_index("table_id").loc[tid, "author_id"] if tid in set(L.load_index().table_id) else tid] += n
        ba = pd.DataFrame([dict(author_id=k, bge_Mtokens=v / 1e6) for k, v in per_work.most_common(25)])
        W.append(md_table(ba))
    else:
        W.append("Not yet counted. Run `python -m src.C.audit --tokens`.")
    W.append("")

    W.append("## 8. Word-by-word (pada) form\n")
    for c, df in samhita_info.items():
        pcols = [x for x in df.columns if x in L.PADA_COLS]
        fill = ", ".join(f"{x}: {100*(df[x].str.strip() != '').mean():.1f}% filled" for x in pcols)
        W.append(f"- {c}: {len(df)} verses. {fill}.")
    W.append("- The pada text (padapatha) is present for every Veda in the samhita tables, in two forms: with accent "
             "marks and without. Words are separated by ` । ` or `।`; compounds are split with `ऽ`.")
    pad = cols_df[(cols_df.gloss == "word-by-word meaning") & (cols_df.role == "text")]
    W.append(f"- Separately, {pad.table_id.nunique()} commentary tables have a word-by-word meaning column "
             "(पदार्थ / Word Meaning / शब्दार्थ): Sanskrit word followed by its meaning. That is a commentary, not the pada text.")
    W.append("")

    W.append("## 9. Assumptions for Stage B (to confirm)\n")
    W.append(ASSUMPTIONS)
    W.append("")

    W.append("## Appendix A. Every column of every table\n")
    c2 = cols_df.copy()
    c2["table_id"] = c2.table_id.str.slice(0, 55)
    W.append(md_table(c2[["table_id", "column", "role", "gloss", "nonempty_pct", "mean_chars", "max_chars", "dup_pct", "scripts", "sample"]]))
    W.append("")
    (FINAL / "data-audit.md").write_text("\n".join(W), encoding="utf-8")
    print("wrote", FINAL / "data-audit.md")


ASSUMPTIONS = """\
1. A *record* is one row of a commentary table: one verse in one work. Its text is the commentary columns (role
   `text` above), cleaned, in sheet order, separated by a blank line, without column names. Verse text, pada text and
   verse metadata columns inside commentary files are not commentary and are left out. Cleaning: placeholder cells
   ('N/A', 'NULL', 'None', '-') become empty, `<br>` becomes a line break, control characters, ZWSP, BOM and soft
   hyphens are removed, NBSP becomes a space, then NFC. ZWJ/ZWNJ are kept.
2. `author_id` = the file name as it is (decided with the human in Stage B): one file is one work; no names are merged.
   Possible same-person pairs are in ../local/final/open-questions.md. The exact commentary columns per table are in
   shared/config.yaml (this audit's 'text' role is the starting point, refined there).
3. The test corpus for retrieval uses Hindi, Sanskrit and English works only (the spec's three languages). Hinglish,
   Bengali, Gujarati, Marathi, Nepali, Tamil and Urdu works are left out of the benchmark corpus but stay in the tables.
4. `multi` anthologies are kept only for rows whose Veda can be inferred without ambiguity (references with 1 part are
   Samaveda, 2 parts Yajurveda, 4 parts Atharvaveda; 3 parts only if the reference exists in just one of Rig/Atharva).
5. The one Marathi cell cut at 32,767 characters is outside the benchmark languages; no action.
6. Samaveda references stored as numbers are read as integers (no fractional part exists).
7. Records with no commentary text, unparseable references or references missing from the samhita are dropped from the
   corpus and counted.
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", action="store_true")
    a = ap.parse_args()
    run(a.tokens)
