"""Verse table with verse details and verse groups.

    python -m src.common.verses      # writes ../local/work/verses.parquet

verse_group: many Samaveda, Yajurveda and Atharvaveda verses repeat Rigveda verses. The group
key is the unaccented verse text after cleaning (NFC, placeholders), with accent marks, spaces,
punctuation and digits removed. Verses with the same key share a group. The group id is the
first verse in the order rig, sama, yajur, atharva, then by reference number.
"""
from __future__ import annotations

import unicodedata

import pandas as pd

from src.common import load as L
from src.common.config import table
from src.common.text import clean

VERSES = L.WORK / "verses.parquet"
COLL_ORDER = {"rig": 0, "sama": 1, "yajur": 2, "atharva": 3}


def _is_accent(ch: str) -> bool:
    o = ord(ch)
    return unicodedata.category(ch) == "Mn" and (0x0951 <= o <= 0x0954 or 0x1CD0 <= o <= 0x1CFF or 0xA8E0 <= o <= 0xA8FF)


def group_key(text: str) -> str:
    t = unicodedata.normalize("NFC", clean(text))
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LM" and not _is_accent(ch))


def ref_sort_key(verse_id: str):
    coll, ref = verse_id.split("/", 1)
    return (COLL_ORDER[coll], tuple(int(x) for x in ref.split("/")))


def build_verses() -> pd.DataFrame:
    rows = []
    for coll in L.SAMHITA_FILES:
        spec = table(f"samhita.{coll}")
        df = L.load_table(f"samhita.{coll}")
        for r in df.to_dict("records"):
            ref = L.norm_ref(r["Mantra Ref ID"])
            if not ref:
                continue
            d = dict(verse_id=f"{coll}/{ref}", collection=coll, ref=ref, excel_row=int(r["_row"]),
                     verse_text=clean(r[spec["verse_text"]]), verse_text_accented=clean(r[spec["verse_text_accented"]]),
                     pada=clean(r[spec["pada"]]), pada_accented=clean(r[spec["pada_accented"]]))
            for c, name in (("Devata", "devata"), ("Rishi", "rishi"), ("Chhanda", "chhanda"),
                            ("Chhand", "chhanda"), ("Swara", "swara")):
                if c in r:
                    d[name] = clean(r[c])
            rows.append(d)
    v = pd.DataFrame(rows)
    v["group_key"] = v.verse_text.map(group_key)
    v = v.iloc[sorted(range(len(v)), key=lambda i: ref_sort_key(v.verse_id.iloc[i]))].reset_index(drop=True)
    first = {}
    for vid, k in zip(v.verse_id, v.group_key):
        if k and k not in first:
            first[k] = vid
    v["verse_group"] = [first[k] if k else vid for vid, k in zip(v.verse_id, v.group_key)]
    v["group_size"] = v.groupby("verse_group").verse_id.transform("size")
    v.drop(columns=["group_key"]).to_parquet(VERSES, index=False)
    return v


def load_verses() -> pd.DataFrame:
    return pd.read_parquet(VERSES) if VERSES.exists() else build_verses()


def group_report(v: pd.DataFrame) -> str:
    g = v.groupby("verse_group")
    multi = v[v.group_size > 1]
    colls = g.collection.agg(lambda x: "+".join(sorted(set(x), key=COLL_ORDER.get)))
    lines = [f"verses: {len(v)}; groups: {g.ngroups}; verses in a group of 2+: {len(multi)} "
             f"({len(multi) / len(v):.1%}); groups of 2+: {int((g.size() > 1).sum())}"]
    for c in COLL_ORDER:
        sub = v[v.collection == c]
        lines.append(f"  {c}: {int((sub.group_size > 1).sum())} of {len(sub)} verses share a group")
    vc = colls[g.size() > 1].value_counts().head(12)
    lines.append("  groups by Vedas involved: " + ", ".join(f"{k} {n}" for k, n in vc.items()))
    within = sum(1 for k, s in g.collection.agg(list).items() if len(s) > 1 and len(set(s)) == 1)
    lines.append(f"  groups of 2+ inside one Veda only (repeated verses): {within}")
    return "\n".join(lines)


if __name__ == "__main__":
    v = build_verses()
    print(group_report(v))
