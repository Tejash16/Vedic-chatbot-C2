"""C5 reranker models (session B).

Each reranker takes one question and its candidate passages and returns a full ranking
(indices into the candidate list, best first) plus the wall time for that question.

- BgeReranker:   BAAI/bge-reranker-v2-m3 cross-encoder, fp16, pair max length 512,
                 only the passage is truncated.
- QwenReranker:  Qwen/Qwen3-Reranker-0.6B, fp16, yes/no logit at the last position
                 (model-card recipe), passage cut to 512 tokens before templating.
- HaikuReranker: claude-haiku-5-5 picks the best 8 of the 50 numbered passages.
                 Its 8 picks come first, then the other candidates in their original order.
"""
import json
import time

import torch

SEED = 7
PASSAGE_MAX_TOKENS = 512


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class BgeReranker:
    name = "BAAI/bge-reranker-v2-m3"

    def __init__(self, device="cuda", batch_size=25, max_length=512):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(SEED)
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.name, torch_dtype=torch.float16).to(device).eval()
        self.device, self.batch_size, self.max_length = device, batch_size, max_length

    @torch.inference_mode()
    def scores(self, question, passages):
        out = []
        for i in range(0, len(passages), self.batch_size):
            part = passages[i:i + self.batch_size]
            enc = self.tok([question] * len(part), part, padding=True, truncation="only_second",
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            out.extend(self.model(**enc).logits.view(-1).float().cpu().tolist())
        return out

    def rank(self, question, passages):
        _sync()
        t0 = time.perf_counter()
        s = self.scores(question, passages)
        _sync()
        ms = (time.perf_counter() - t0) * 1000.0
        order = sorted(range(len(passages)), key=lambda j: (-s[j], j))
        return order, ms, s

    def close(self):
        del self.model
        torch.cuda.empty_cache()


QWEN_INSTRUCTION = ("Given a question about a Vedic mantra, possibly in Hindi, English, Romanized "
                    "Hindi or quoting Sanskrit words, judge whether the commentary passage answers it")


class QwenReranker:
    name = "Qwen/Qwen3-Reranker-0.6B"
    prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
              "Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
              "<|im_end|>\n<|im_start|>user\n")
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(self, device="cuda", batch_size=10, dtype=torch.float16,
                 instruction=QWEN_INSTRUCTION):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(SEED)
        self.tok = AutoTokenizer.from_pretrained(self.name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name, torch_dtype=dtype).to(device).eval()
        self.yes = self.tok.convert_tokens_to_ids("yes")
        self.no = self.tok.convert_tokens_to_ids("no")
        self.pre = self.tok.encode(self.prefix, add_special_tokens=False)
        self.suf = self.tok.encode(self.suffix, add_special_tokens=False)
        self.device, self.batch_size, self.instruction = device, batch_size, instruction

    def _cut(self, text):
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= PASSAGE_MAX_TOKENS:
            return text
        return self.tok.decode(ids[:PASSAGE_MAX_TOKENS])

    @torch.inference_mode()
    def scores(self, question, passages):
        out = []
        texts = ["<Instruct>: %s\n<Query>: %s\n<Document>: %s" % (self.instruction, question, self._cut(p))
                 for p in passages]
        for i in range(0, len(texts), self.batch_size):
            part = texts[i:i + self.batch_size]
            enc = self.tok(part, padding=False, truncation=False, add_special_tokens=False,
                           return_attention_mask=False)
            enc["input_ids"] = [self.pre + ids + self.suf for ids in enc["input_ids"]]
            enc = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
            logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            out.extend(torch.log_softmax(pair, dim=1)[:, 1].exp().cpu().tolist())
        return out

    def rank(self, question, passages):
        _sync()
        t0 = time.perf_counter()
        s = self.scores(question, passages)
        _sync()
        ms = (time.perf_counter() - t0) * 1000.0
        order = sorted(range(len(passages)), key=lambda j: (-s[j], j))
        return order, ms, s

    def close(self):
        del self.model
        torch.cuda.empty_cache()


HAIKU_MODEL = "claude-haiku-5-5"
HAIKU_SYSTEM = (
    "You rerank search results for a question-answering assistant over Vedic mantras and their "
    "commentaries in Sanskrit, Hindi and English. Questions may be in Hindi, English, Hindi typed in "
    "Roman letters, or may quote Sanskrit words from a mantra. You are given one question and numbered "
    "commentary passages, each with a header giving the verse reference, author and collection. Choose "
    "the 8 passages most likely to contain the answer, best first. Judge only by whether the passage "
    "answers the question; a passage about the right verse that does not address the question ranks "
    "below one that does.")
HAIKU_SCHEMA = {
    "type": "object",
    "properties": {"best": {"type": "array", "items": {"type": "integer"}}},
    "required": ["best"],
    "additionalProperties": False,
}


def haiku_user_message(question, passages):
    parts = ["Question:\n%s\n\nPassages:" % question]
    for i, p in enumerate(passages, 1):
        parts.append("[%d]\n%s" % (i, p))
    parts.append("Return JSON {\"best\": [...]} with the numbers of the 8 best passages, best first.")
    return "\n\n".join(parts)


def haiku_request(question, passages, effort="medium", max_tokens=8000):
    return dict(
        model=HAIKU_MODEL,
        max_tokens=max_tokens,
        system=HAIKU_SYSTEM,
        messages=[{"role": "user", "content": haiku_user_message(question, passages)}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": HAIKU_SCHEMA}},
    )


def haiku_parse(message, n):
    """Picks (0-based, deduplicated, in range) from a Messages API response, plus a status."""
    if message.stop_reason == "refusal":
        return [], "refusal"
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        best = json.loads(text)["best"]
    except Exception:
        return [], "bad_json"
    picks = []
    for x in best:
        if isinstance(x, int) and 1 <= x <= n and (x - 1) not in picks:
            picks.append(x - 1)
    return picks[:8], ("ok" if len(picks) >= 8 else "short")


def haiku_order(picks, n):
    rest = [j for j in range(n) if j not in picks]
    return list(picks) + rest
