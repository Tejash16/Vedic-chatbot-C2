"""bge-m3 dense embeddings with an on-disk cache in ../local/work/D, plus a fake embedder
for dry runs. Half precision, explicit maximum lengths."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np

from dutil import WORK_D

PASSAGE_MAXLEN = 512
QUERY_MAXLEN = 128


def _key(texts, tag):
    h = hashlib.sha256(tag.encode())
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    return h.hexdigest()[:24]


def local_snapshot(repo="BAAI/bge-m3") -> str:
    """Path of the cached model snapshot. Passing a path stops FlagEmbedding from calling
    snapshot_download (which fails on Windows symlinks and would write to the shared cache)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, local_files_only=True)


class BGEM3:
    name = "BAAI/bge-m3"

    def __init__(self, batch_size=32):
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        from FlagEmbedding import BGEM3FlagModel
        import torch
        self.torch = torch
        self.model = BGEM3FlagModel(local_snapshot(), use_fp16=True,
                                    devices="cuda:0" if torch.cuda.is_available() else "cpu")
        self.bs = batch_size

    def encode(self, texts, max_length):
        out = self.model.encode(list(texts), batch_size=self.bs, max_length=max_length,
                                return_dense=True, return_sparse=False, return_colbert_vecs=False)
        v = np.asarray(out["dense_vecs"], dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)

    def close(self):
        try:
            self.model.stop_self_pool()
        except Exception:
            pass
        del self.model
        import gc
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class LazyBGEM3:
    """Loads bge-m3 only on the first cache miss, so re-running filters on cached embeddings
    needs no GPU."""
    name = "BAAI/bge-m3"

    def __init__(self):
        self._m = None

    def encode(self, texts, max_length):
        if self._m is None:
            self._m = BGEM3()
        return self._m.encode(texts, max_length)

    def close(self):
        if self._m is not None:
            self._m.close()
            self._m = None


class FakeEmbedder:
    """Hashed character-trigram bag. Deterministic, CPU, no model download."""
    name = "fake-trigram"

    def __init__(self, dim=512):
        self.dim = dim

    def encode(self, texts, max_length):
        from dutil import norm
        m = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            s = " %s " % norm(t).casefold()
            for j in range(len(s) - 2):
                h = int(hashlib.md5(s[j:j + 3].encode()).hexdigest()[:8], 16)
                m[i, h % self.dim] += 1.0
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def close(self):
        pass


def cached_encode(emb, texts, max_length, tag, cache_dir: Path = WORK_D / "emb"):
    """Encode with a cache keyed on model, max length and the exact texts."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / ("%s_%s_%d_%s.npy" % (tag, emb.name.replace("/", "_"), max_length,
                                            _key(texts, emb.name + str(max_length))))
    if path.exists():
        return np.load(path)
    t0 = time.time()
    v = emb.encode(texts, max_length)
    np.save(path, v)
    print("encoded %d texts (%s, max_length=%d) in %.1fs" % (len(texts), tag, max_length, time.time() - t0))
    return v
