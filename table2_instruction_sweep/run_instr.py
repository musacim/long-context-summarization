import argparse, glob, json, os
import requests
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.join(HERE, "..", "table1_length_curve")
QA_PROMPT = open(os.path.join(T1, "prompts", "qa_neutral.txt")).read()
BIN = 100000

CONDS = [
    ("none", None),
    ("tok2000", "at least 2,000 tokens long"),
    ("tok4000", "at least 4,000 tokens long"),
    ("tok8000", "at least 8,000 tokens long"),
    ("tok16000", "at least 16,000 tokens long"),
    ("w1250", "at least 1,250 words long"),
    ("w2500", "at least 2,500 words long"),
    ("w5000", "at least 5,000 words long"),
    ("w9950", "at least 9,950 words long"),
    ("c8900", "at least 8,900 characters long"),
    ("c17800", "at least 17,800 characters long"),
    ("c35600", "at least 35,600 characters long"),
    ("c71200", "at least 71,200 characters long"),
    ("rel_quarter", "at least one quarter as long as the document"),
    ("rel_third", "at least one third as long as the document"),
    ("rel_half", "at least half as long as the document"),
]
CMAP = dict(CONDS)

def condenser_prompt(doc):
    t = open(os.path.join(T1, "prompts", "openhands_condenser.j2")).read()
    head, _, _ = t.partition("{% for event in events %}")
    return head + "<EVENT>\n" + doc + "\n</EVENT>\n\nNow summarize the events using the rules above."

def build_prompt(domain, text, phrase):
    if domain == "qa":
        if phrase is None:
            return QA_PROMPT + text
        return f"Summarize the following document. Your summary must be {phrase}.\n\n" + text
    p = condenser_prompt(text)
    if phrase is None:
        return p
    return p + f" Your summary must be {phrase}."

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--domain", choices=["qa", "agentic"], required=True)
    ap.add_argument("--docs", required=True)
    ap.add_argument("--ndocs", type=int, default=3)
    ap.add_argument("--conds", default="all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args()

    keys = [k for k, _ in CONDS] if a.conds == "all" else a.conds.split(",")
    tok = Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    docs = [json.loads(l) for l in open(a.docs)][: a.ndocs]
    for d in docs:
        ids = tok.encode(d["doc"]).ids
        text = tok.decode(ids[:BIN])
        for k in keys:
            cdir = os.path.join(a.model_dir, a.domain, k)
            os.makedirs(cdir, exist_ok=True)
            out = os.path.join(cdir, f"results{a.tag}.jsonl")
            done = set()
            for rf in glob.glob(os.path.join(cdir, "results*.jsonl")):
                done |= {json.loads(l)["_id"] for l in open(rf)}
            if d["_id"] in done:
                continue
            prompt = build_prompt(a.domain, text, CMAP[k])
            r = requests.post(f"{a.url}/chat/completions", json={
                "model": a.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": a.max_tokens, "n": a.reps,
                "temperature": 0.7, "top_p": 0.9, "repetition_penalty": 1.05,
            }, timeout=3600).json()
            if "choices" not in r:
                print(f"  ERROR {d['_id']} {k}: {str(r)[:200]}", flush=True)
                continue
            reps = []
            for c in r["choices"]:
                msg = c["message"]
                body = msg.get("content") or ""
                think = msg.get("reasoning_content") or msg.get("reasoning") or ""
                rep = {"n_tok": len(tok.encode(body).ids),
                       "n_words": len(body.split()), "n_chars": len(body),
                       "finish_reason": c["finish_reason"], "text": body}
                if think:
                    rep["think_tok"] = len(tok.encode(think).ids)
                reps.append(rep)
            with open(out, "a") as f:
                f.write(json.dumps({"_id": d["_id"], "cond": k, "reps": reps}) + "\n")
            print(f"  {d['_id']} {k}: " +
                  " ".join(f"{x['n_tok']}({x['finish_reason'][:3]})" for x in reps), flush=True)

main()
