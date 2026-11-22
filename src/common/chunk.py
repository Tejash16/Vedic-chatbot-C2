"""Records, chunks and the test corpus. Same Excel files in, same chunks out, on any laptop.

    python -m src.common.chunk                  # Excel -> tables -> records -> test corpus chunks
                                                # (../local/work/chunks.parquet), checked against
                                                # shared/chunks_manifest.json
    python -m src.common.chunk --target 150     # C6: same records, other chunk size
                                                # (../local/work/chunks_150.parquet)
    python -m src.common.chunk --full           # every hi/sa/en record (../local/work/chunks_full.parquet)
    python -m src.common.chunk --write-manifest # session C only

Commentary columns come from shared/config.yaml (tables.<id>.commentary). A record's text is
those columns, cleaned, in sheet order, joined by a blank line. Chunks break at a column
boundary first, then at sentence ends. `sections` lists the columns a chunk covers. Neither the
text nor the header holds column names. Embed with embed_text(row, header=True).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import re
import sys

import pandas as pd

from src.common import load as L
from src.common.config import config
from src.common.text import FIELD_SEP, clean, sentence_spans

SEED = 7
CORPUS_LANGS = ("hi", "sa", "en")
TARGET_TOKENS = 300
CORPUS_CHUNKS = 10_000
SMALL = 0.3          # a chunk below SMALL x target is "small"
TAIL_MAX = 1.3       # a small last chunk may join the one before up to TAIL_MAX x target
TOKENIZER = "BAAI/bge-m3"
MANIFEST = L.ROOT / "shared" / "chunks_manifest.json"
RECORDS = L.WORK / "records.parquet"

VEDA = {"rig": "Rigveda", "sama": "Samaveda", "yajur": "Yajurveda", "atharva": "Atharvaveda"}
VEDA_ORDER = ("rig", "sama", "yajur", "atharva")

_tok = None


def tokenizer():
    global _tok
    if _tok is None:
        import transformers
        from transformers import AutoTokenizer
        transformers.logging.set_verbosity_error()
        _tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return _tok


def h(s: str, n: int = 16) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def work_name(stem: str) -> str:
    """'RigvedaBhashyaAryamuniHindi' -> 'Rigveda Bhashya Aryamuni Hindi' (the file name, spaced)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)


# ---------------------------------------------------------------- records
def record_sections(row: dict, spec: dict) -> list[tuple[str, str, str]]:
    """(column, language, cleaned text) for each non-empty commentary column, in sheet order."""
    out = []
    rules = spec.get("cell_rules", {})
    langs = spec.get("column_lang", {})
    for c in spec["commentary"]:
        v = clean(row.get(c, ""))
        if v and c in rules and re.search(rules[c]["drop_if_matches"], v):
            v = ""
        if v:
            out.append((c, langs.get(c, spec["lang"]), v))
    return out


def build_records() -> pd.DataFrame:
    """Every commentary record (all languages) whose reference points to exactly one verse and
    which has commentary text. Section spans are character offsets into `text`."""
    if not (L.TABLES / "_index.parquet").exists():
        L.convert_all(verbose=False)
    tables = config()["tables"]
    samhita = {}
    for coll in L.SAMHITA_FILES:
        df = L.load_table(f"samhita.{coll}")
        samhita[coll] = set(x for x in df["Mantra Ref ID"].map(L.norm_ref) if x)
    out = []
    for tid, spec in tables.items():
        if spec["kind"] != "bhashya" or not spec["commentary"]:
            continue
        df = L.load_table(tid)
        for row in df.to_dict("records"):
            ref = L.norm_ref(row["Mantra Ref ID"])
            if not ref:
                continue
            cands = L.ref_collection(ref) if spec["collection"] == "multi" else [spec["collection"]]
            hits = [c for c in cands if ref in samhita[c]]
            if len(hits) != 1:
                continue
            secs = record_sections(row, spec)
            if not secs:
                continue
            parts, starts, ends, pos = [], [], [], 0
            for _, _, t in secs:
                if parts:
                    pos += len(FIELD_SEP)
                starts.append(pos)
                pos += len(t)
                ends.append(pos)
                parts.append(t)
            out.append(dict(record_id=f"{tid}#{row['_row']}", table_id=tid, author_id=spec["author_id"],
                            verse_id=f"{hits[0]}/{ref}", collection=hits[0], ref=ref, lang=spec["lang"],
                            text=FIELD_SEP.join(parts), sec_names=[s[0] for s in secs],
                            sec_langs=[s[1] for s in secs], sec_start=starts, sec_end=ends))
    rec = pd.DataFrame(out).sort_values("record_id", kind="stable").reset_index(drop=True)
    rec.to_parquet(RECORDS, index=False)
    return rec


def load_records() -> pd.DataFrame:
    return pd.read_parquet(RECORDS) if RECORDS.exists() else build_records()


# ---------------------------------------------------------------- chunking
def _units(text: str, a0: int, b0: int, ntok, target: int) -> list[tuple[int, int]]:
    """Sentence spans inside [a0, b0); a sentence longer than target is cut at spaces."""
    units = []
    for a, b in sentence_spans(text[a0:b0]):
        a, b = a + a0, b + a0
        if ntok(a, b) <= target:
            units.append((a, b))
            continue
        s = a
        while s < b:
            best = s
            for i in range(s + 1, b + 1):
                if i == b or text[i].isspace():
                    if ntok(s, i) <= target:
                        best = i
                    else:
                        break
            e = best if best > s else b
            units.append((s, e))
            s = e
            while s < b and text[s].isspace():
                s += 1
    return units


def chunk_record(text: str, secs: list[tuple[str, str, int, int]], target: int = TARGET_TOKENS):
    """secs: (column, language, start, end). Returns (start, end, n_tokens, columns, language)."""
    enc = tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    tstarts = [a for a, b in enc["offset_mapping"] if b > a]

    def ntok(a, b):
        return bisect.bisect_left(tstarts, b) - bisect.bisect_left(tstarts, a)

    chunks = []          # [start, end, [columns], lang]
    cur = None

    def emit():
        nonlocal cur
        if cur:
            chunks.append(cur)
        cur = None

    for col, lang, a, b in secs:
        if cur and cur[3] == lang and ntok(cur[0], b) <= target:          # whole column fits
            cur[1] = b
            cur[2].append(col)
            continue
        if cur and (cur[3] != lang or ntok(cur[0], cur[1]) >= SMALL * target):
            emit()                                                        # break at the column boundary
        if cur is None and ntok(a, b) <= target:
            cur = [a, b, [col], lang]
            continue
        for ua, ub in _units(text, a, b, ntok, target):                   # then at sentence ends
            if cur is None:
                cur = [ua, ub, [col], lang]
            elif ntok(cur[0], ub) <= target:
                cur[1] = ub
                if cur[2][-1] != col:
                    cur[2].append(col)
            else:
                emit()
                cur = [ua, ub, [col], lang]
    emit()
    if (len(chunks) >= 2 and chunks[-1][3] == chunks[-2][3] and ntok(chunks[-1][0], chunks[-1][1]) < SMALL * target
            and ntok(chunks[-2][0], chunks[-1][1]) <= TAIL_MAX * target):
        last = chunks.pop()
        chunks[-1][1] = last[1]
        chunks[-1][2] += [c for c in last[2] if c != chunks[-1][2][-1]]
    return [(a, b, ntok(a, b), cols, lang) for a, b, cols, lang in chunks]


def header_of(r) -> str:
    return f"{VEDA[r['collection']]} {r['ref']} | {work_name(r['author_id'])}"


def chunk_records(rec: pd.DataFrame, target: int = TARGET_TOKENS) -> pd.DataFrame:
    rows = []
    for r in rec.to_dict("records"):
        secs = list(zip(r["sec_names"], r["sec_langs"], r["sec_start"], r["sec_end"]))
        hdr = header_of(r)
        for a, b, n, cols, lang in chunk_record(r["text"], secs, target):
            t = r["text"][a:b]
            rows.append(dict(chunk_id=h(f"{r['record_id']}|{a}|{b}|{t}"), record_id=r["record_id"],
                             verse_id=r["verse_id"], author_id=r["author_id"], collection=r["collection"],
                             lang=lang, char_start=a, char_end=b, n_tokens=n, header=hdr, text=t,
                             sections=list(cols)))
    return pd.DataFrame(rows)


def embed_text(row, header: bool = True) -> str:
    return f"{row['header']}\n{row['text']}" if header else row["text"]


# ---------------------------------------------------------------- test corpus
def verse_order(verse_ids) -> list[str]:
    """Data-only order that spreads the Vedas evenly: within each Veda, rank verses by
    sha256('7|' + verse_id); then interleave all Vedas by rank / (verses of that Veda)."""
    by = {}
    for v in sorted(set(verse_ids)):
        by.setdefault(v.split("/")[0], []).append(v)
    keyed = []
    for coll, vs in by.items():
        vs = sorted(vs, key=lambda x: h(f"{SEED}|{x}"))
        n = len(vs)
        keyed += [((i + 0.5) / n, h(f"{SEED}|{x}"), x) for i, x in enumerate(vs)]
    return [x for _, _, x in sorted(keyed)]


def select_corpus(rec: pd.DataFrame, n_chunks: int = CORPUS_CHUNKS) -> pd.DataFrame:
    el = rec[rec.lang.isin(CORPUS_LANGS)]
    by_verse = {v: g for v, g in el.groupby("verse_id", sort=False)}
    parts, total = [], 0
    for v in verse_order(el.verse_id):
        ch = chunk_records(by_verse[v].sort_values("record_id"))
        parts.append(ch)
        total += len(ch)
        if total >= n_chunks:
            break
    return pd.concat(parts, ignore_index=True)


def manifest_of(ch: pd.DataFrame) -> dict:
    import tokenizers
    import transformers
    ids = sorted(ch.chunk_id)
    return dict(n_chunks=len(ids), n_records=int(ch.record_id.nunique()), n_verses=int(ch.verse_id.nunique()),
                sha256_sorted_chunk_ids=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                target_tokens=TARGET_TOKENS, tokenizer=TOKENIZER, seed=SEED, langs=list(CORPUS_LANGS),
                transformers=transformers.__version__, tokenizers=tokenizers.__version__,
                rebuild="python -m src.common.chunk --rebuild",
                excel_sha256={r.path.replace("\\", "/"): r.sha256 for r in L.load_index().itertuples()})


def chunks_path(target: int = TARGET_TOKENS, full: bool = False):
    if full:
        return L.WORK / "chunks_full.parquet"
    return L.WORK / ("chunks.parquet" if target == TARGET_TOKENS else f"chunks_{target}.parquet")


def load_chunks(target: int = TARGET_TOKENS, full: bool = False) -> pd.DataFrame:
    return pd.read_parquet(chunks_path(target, full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=TARGET_TOKENS)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--rebuild", action="store_true", help="rebuild tables and records from the Excel files")
    a = ap.parse_args()
    if a.rebuild or not (L.TABLES / "_index.parquet").exists():
        L.convert_all(verbose=False)
    rec = build_records() if (a.rebuild or not RECORDS.exists()) else load_records()
    if a.full:
        ch = chunk_records(rec[rec.lang.isin(CORPUS_LANGS)], a.target)
        ch.to_parquet(chunks_path(full=True), index=False)
        print(f"wrote {chunks_path(full=True)}: {len(ch)} chunks")
        return
    if a.target == TARGET_TOKENS:
        ch = select_corpus(rec)
        ch.to_parquet(chunks_path(), index=False)
        m = manifest_of(ch)
        print(json.dumps(m, indent=1))
        if a.write_manifest:
            MANIFEST.parent.mkdir(exist_ok=True)
            MANIFEST.write_text(json.dumps(m, indent=1) + "\n", encoding="utf-8")
            print("wrote", MANIFEST)
        elif MANIFEST.exists():
            ref = json.loads(MANIFEST.read_text(encoding="utf-8"))
            ok = ref["n_chunks"] == m["n_chunks"] and ref["sha256_sorted_chunk_ids"] == m["sha256_sorted_chunk_ids"]
            print("MANIFEST MATCH" if ok else "MANIFEST MISMATCH")
            sys.exit(0 if ok else 1)
    else:
        base = load_chunks()
        ch = chunk_records(rec[rec.record_id.isin(set(base.record_id))].sort_values("record_id"), a.target)
        ch.to_parquet(chunks_path(a.target), index=False)
        print(f"wrote {chunks_path(a.target)}: {len(ch)} chunks from {ch.record_id.nunique()} records")


if __name__ == "__main__":
    main()
