"""C1b helper: OpenSearch container, index and queries (session B).

OpenSearch runs in Docker on this laptop, security plugin off, bound to localhost only.
Three text fields per chunk:
  text_hi    Lucene 'hindi' analyzer   (filled for Hindi and Sanskrit chunks)
  text_en    Lucene 'english' analyzer (filled for English chunks)
  text_plain standard tokenizer + lowercase, no stemming, no stopwords (filled for every chunk)
BM25 with the OpenSearch defaults (k1 = 1.2, b = 0.75).
"""
import subprocess
import time
import unicodedata

from opensearchpy import OpenSearch, helpers

IMAGE = "opensearchproject/opensearch:2.19.1"
CONTAINER = "c2-opensearch-b"
PORT = 9200
INDEX = "c2_chunks_b"
HEAP = "768m"

SETTINGS = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "-1",
        "analysis": {
            "analyzer": {
                "plain": {"type": "custom", "tokenizer": "standard", "filter": ["lowercase"]},
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "chunk_id": {"type": "keyword"},
            "verse_id": {"type": "keyword"},
            "lang": {"type": "keyword"},
            "text_hi": {"type": "text", "analyzer": "hindi"},
            "text_en": {"type": "text", "analyzer": "english"},
            "text_plain": {"type": "text", "analyzer": "plain"},
        },
    },
}


def start_container():
    """Start (or reuse) the OpenSearch container and wait until the cluster answers."""
    r = subprocess.run(["docker", "ps", "-a", "--filter", "name=^/%s$" % CONTAINER,
                        "--format", "{{.Status}}"], capture_output=True, text=True)
    status = r.stdout.strip()
    if not status:
        subprocess.run([
            "docker", "run", "-d", "--name", CONTAINER,
            "-p", "127.0.0.1:%d:9200" % PORT,
            "-e", "discovery.type=single-node",
            "-e", "DISABLE_SECURITY_PLUGIN=true",
            "-e", "DISABLE_INSTALL_DEMO_CONFIG=true",
            "-e", "bootstrap.memory_lock=false",
            "-e", "OPENSEARCH_JAVA_OPTS=-Xms%s -Xmx%s" % (HEAP, HEAP),
            IMAGE], check=True)
    elif not status.startswith("Up"):
        subprocess.run(["docker", "start", CONTAINER], check=True)
    client = OpenSearch(hosts=[{"host": "127.0.0.1", "port": PORT}], use_ssl=False,
                        timeout=120, http_compress=False)
    deadline = time.time() + 600  # startup takes ~3 minutes when free RAM is low
    last = None
    while time.time() < deadline:
        try:
            h = client.cluster.health(wait_for_status="yellow", request_timeout=10)
            if h.get("status") in ("yellow", "green"):
                return client
        except Exception as e:  # not up yet
            last = e
        time.sleep(3)
    raise RuntimeError("OpenSearch did not come up on port %d: %r" % (PORT, last))


def stop_container():
    subprocess.run(["docker", "stop", CONTAINER], check=False)


def server_version(client):
    return client.info()["version"]["number"]


def build_index(client, rows, recreate=True):
    """rows: iterable of dicts with chunk_id, verse_id, lang, route ('hi' or 'en'), text."""
    if recreate and client.indices.exists(index=INDEX):
        client.indices.delete(index=INDEX)
    client.indices.create(index=INDEX, body=SETTINGS)

    def actions():
        for r in rows:
            doc = {"chunk_id": r["chunk_id"], "verse_id": r["verse_id"], "lang": r["lang"],
                   "text_plain": r["text"]}
            doc["text_hi" if r["route"] == "hi" else "text_en"] = r["text"]
            yield {"_index": INDEX, "_id": r["chunk_id"], "_source": doc}

    ok, errors = helpers.bulk(client, actions(), chunk_size=500, request_timeout=300,
                              raise_on_error=False)
    client.indices.put_settings(index=INDEX, body={"index": {"refresh_interval": "1s"}})
    client.indices.refresh(index=INDEX)
    client.indices.forcemerge(index=INDEX, max_num_segments=1, request_timeout=600)
    client.indices.refresh(index=INDEX)
    n = client.count(index=INDEX)["count"]
    return ok, errors, n


QUERIES = {
    # language-analysed fields, question sent to the field of its own script
    # (Devanagari -> Hindi analyzer field, Latin -> English analyzer field)
    "lang_routed": lambda q: {"match": {("text_hi" if is_devanagari(q) else "text_en"): {"query": q}}},
    # both language fields, best raw BM25 score wins (kept as a negative result: scores from
    # fields with different statistics are not comparable)
    "lang": lambda q: {"multi_match": {"query": q, "fields": ["text_hi", "text_en"],
                                       "type": "best_fields"}},
    # unstemmed field only
    "plain": lambda q: {"match": {"text_plain": {"query": q}}},
    # one query over all three fields, scores summed
    "all_mf": lambda q: {"multi_match": {"query": q, "fields": ["text_hi", "text_en", "text_plain"],
                                         "type": "most_fields"}},
}


def is_devanagari(text):
    """True when the question has more Devanagari letters than Latin letters."""
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ" and c.isalpha())
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    return dev > lat


def search(client, kind, question, k=50):
    """Return (chunk_ids, wall_ms, server_took_ms)."""
    q = unicodedata.normalize("NFC", question)
    body = {"size": k, "_source": False, "query": QUERIES[kind](q)}
    t0 = time.perf_counter()
    res = client.search(index=INDEX, body=body, request_timeout=60)
    wall = (time.perf_counter() - t0) * 1000.0
    return [h["_id"] for h in res["hits"]["hits"]], wall, float(res.get("took", 0))


def analyze(client, analyzer, text):
    """Tokens produced by a built-in or index analyzer (for checks only)."""
    res = client.indices.analyze(index=INDEX, body={"analyzer": analyzer, "text": text})
    return [t["token"] for t in res["tokens"]]
