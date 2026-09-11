import argparse, json, os, re
import requests
from tokenizers import Tokenizer

BINS = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 100000]
HERE = os.path.dirname(os.path.abspath(__file__))
QA_PROMPT = open(os.path.join(HERE, "prompts", "qa_neutral.txt")).read()

def condenser_prompt(doc):
    t = open(os.path.join(HERE, "prompts", "openhands_condenser.j2")).read()
    head, _, _ = t.partition("{% for event in events %}")
    return head + "<EVENT>\n" + doc + "\n</EVENT>\n\nNow summarize the events using the rules above."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--domain", choices=["qa", "agentic"], required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--bins", default="")
    a = ap.parse_args()

    tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    docs = [json.loads(l) for l in open(a.docs)]
    for d in docs:
        ids = tok.encode(d["doc"]).ids
        bins = [int(x)*1000 for x in a.bins.split(",")] if a.bins else BINS
        for b in bins:
            bdir = os.path.join(a.model_dir, a.domain, f"{b//1000}k")
            os.makedirs(bdir, exist_ok=True)
            out = os.path.join(bdir, f"results{a.tag}.jsonl")
            done = set()
            import glob as _g
            for rf in _g.glob(os.path.join(bdir, "results*.jsonl")):
                done |= {json.loads(l)["_id"] for l in open(rf)}
            if d["_id"] in done:
                continue
            text = tok.decode(ids[:b])
            prompt = QA_PROMPT + text if a.domain == "qa" else condenser_prompt(text)
            r = requests.post(f"{a.url}/chat/completions", json={
                "model": a.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "n": a.reps,
                "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05,
            }, timeout=3600).json()
            if "choices" not in r:
                print(f"  ERROR {d['_id']} {b}: {str(r)[:200]}", flush=True)
                continue
            reps = []
            for c in r["choices"]:
                msg = c["message"]
                body = msg.get("content") or ""
                think = msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not think and "</think>" in body:
                    think, body = body.split("</think>", 1)
                    think = think.replace("<think>", "").strip()
                    body = body.strip()
                rep = {"n_tok": len(tok.encode(body).ids),
                       "finish_reason": c["finish_reason"], "text": body}
                if think:
                    rep["think_tok"] = len(tok.encode(think).ids)
                reps.append(rep)
            with open(out, "a") as f:
                f.write(json.dumps({"_id": d["_id"], "bin": b, "reps": reps}) + "\n")
            print(f"  {d['_id']} {b//1000}k: " +
                  " ".join(f"{x['n_tok']}({x['finish_reason'][:3]})" for x in reps), flush=True)

main()
