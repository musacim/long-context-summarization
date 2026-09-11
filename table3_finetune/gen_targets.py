import argparse, json, os, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from tokenizers import Tokenizer

TOK = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
ntok = lambda s: len(TOK.encode(s).ids)

INSTR = {
 "medium": ("Summarize the following document in about 400 tokens (~300 words). "
    "Write a single, complete, self-contained prose summary of the document's "
    "content. No preamble, no meta-commentary, no bullet lists, no headers. "
    "Begin directly with the summary and end on a complete sentence.\n\nDOCUMENT:\n"),
 "short": ("Rewrite the following summary as a much shorter summary of about 100 tokens "
    "(~75 words). Keep the most important content. It must read as a single, "
    "complete, self-contained prose summary in its own right - not a truncated "
    "fragment. No preamble, no meta-commentary, no bullet lists, no headers. "
    "Begin directly with the summary and end on a complete sentence.\n\nSUMMARY:\n"),
 "long": ("Write a long, detailed summary of the following document. "
    "It must be AT LEAST 2000 tokens (roughly 1500 words). "
    "Cover the full arc of the document in order: every major character, event, and development. "
    "Write flowing connected prose as a single continuous summary. "
    "No bullet lists, no headers, no meta-commentary. Do not stop early.\n\nDOCUMENT:\n"),
}
CHAT = ("let me know", "would you like", "which direction", "i can also",
        "happy to", "if you'd like", "shall i", "let me help")

def call_teacher(prompt, model="sonnet", timeout=1200):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(prompt); path = f.name
    try:
        with open(path) as fh:
            r = subprocess.run(["claude", "-p", "--model", model], stdin=fh,
                               capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode, r.stderr.strip()[:200]
    finally:
        os.unlink(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["medium", "short", "long"])
    ap.add_argument("--docs", required=True,
                    help="train_docs.jsonl for medium/long; the MEDIUM summaries file for short")
    ap.add_argument("--out", required=True)
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--teacher", default="sonnet")
    a = ap.parse_args()

    key = "summary" if a.mode == "short" else "doc"
    src = [json.loads(l) for l in open(a.docs)][a.start:a.start + a.n]
    done = set()
    if os.path.exists(a.out):
        done = {json.loads(l)["_id"] for l in open(a.out) if "error" not in json.loads(l)}
    todo = [d for d in src if d["_id"] not in done and "error" not in d]
    print(f"{len(todo)} to generate ({len(done)} done)", flush=True)

    def bad_long(txt, n):
        low = txt[-300:].lower()
        return n < 1200 or any(c in low for c in CHAT)

    def work(d):
        tries = 3 if a.mode == "long" else 1
        raw, n = "", 0
        for _ in range(tries):
            raw, rc, err = call_teacher(INSTR[a.mode] + d[key], a.teacher)
            if rc != 0 or not raw:
                raw = ""; continue
            n = ntok(raw)
            if a.mode != "long" or not bad_long(raw, n):
                return {"_id": d["_id"], "n_tok": n, "summary": raw}
        if not raw:
            return {"_id": d["_id"], "error": err or "empty", "n_tok": 0}
        return {"_id": d["_id"], "n_tok": n, "summary": raw, "flag": "short_or_chatty"}

    with open(a.out, "a") as out, ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, todo):
            out.write(json.dumps(r) + "\n"); out.flush()
            print(f"  {r['_id']}: {r.get('error') or str(r['n_tok'])+' tok'}", flush=True)

main()
