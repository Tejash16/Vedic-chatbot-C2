"""Qwen/Qwen3-Embedding-0.6B dense encoder for step C2b (session A).

Half precision, max length 512 for passages, L2-normalised vectors.
Queries use the model's own "query" prompt (an instruction prefix); passages
use no prompt, as the model card recommends.
Embeddings are cached as .npy under ../local/work/A so reruns are free.
"""
import gc
import hashlib
import os
import time
from pathlib import Path

import numpy as np

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
SEED = 7

ROOT = Path(__file__).resolve().parents[2]          # the clone
LOCAL = ROOT.parent / "local"
CACHE = LOCAL / "work" / "A"
os.environ.setdefault("HF_HOME", str(LOCAL / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def local_path(model_id):
    from huggingface_hub import snapshot_download
    return snapshot_download(model_id, local_files_only=True)


def _key(texts, tag):
    h = hashlib.sha256()
    h.update(tag.encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


class QwenEncoder:
    def __init__(self, max_len=512, batch_size=16, device="cuda"):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.manual_seed(SEED)
        self.torch = torch
        self.model = SentenceTransformer(
            local_path(MODEL_ID), device=device,
            model_kwargs={"dtype": torch.float16},
            processor_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = max_len
        self.batch_size = batch_size

    def encode(self, texts, is_query=False, batch_size=None):
        kw = {"prompt_name": "query"} if is_query else {}
        with self.torch.inference_mode():
            v = self.model.encode(
                list(texts), batch_size=batch_size or self.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=len(texts) > 500, **kw)
        return v.astype(np.float16)

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()


def cached_encode(enc_factory, texts, tag, is_query=False):
    """Encode with an on-disk cache keyed by the texts. enc_factory() is only
    called on a cache miss, so a cached run never touches the GPU."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / ("qwen3_%s_%s.npy" % (tag, _key(texts, tag)))
    side = path.with_suffix(".secs")                  # encoding time of the run that built the cache
    if path.exists():
        return np.load(path), (float(side.read_text()) if side.exists() else 0.0), None
    enc = enc_factory()
    t0 = time.perf_counter()
    v = enc.encode(texts, is_query=is_query)
    secs = time.perf_counter() - t0
    np.save(path, v)
    side.write_text(f"{secs:.1f}")
    return v, secs, enc
