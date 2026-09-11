import argparse, json, statistics as st, requests

NEUTRAL = "Summarize the following document.\n\n"

ap = argparse.ArgumentParser()
ap.add_argument("--url", required=True)
ap.add_argument("--held-out", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--models", nargs="+", required=True, metavar="label=served_name")
ap.add_argument("--max-tokens", type=int, default=3072)
ap.add_argument("-n", type=int, default=20)
a = ap.parse_args()
models = [m.split("=", 1) for m in a.models]

def gen(served, doc):
    r = requests.post(f"{a.url}/chat/completions", json={
        "model": served, "messages": [{"role": "user", "content": NEUTRAL + doc}],
        "max_tokens": a.max_tokens, "temperature": 0.0}, timeout=900)
    j = r.json()
    return j["usage"]["completion_tokens"], j["choices"][0]["message"]["content"]

rows = [json.loads(l) for l in open(a.held_out)][:a.n]
res = {lab: [] for lab, _ in models}
out = []
for i, row in enumerate(rows):
    rec = {"i": i, "_id": row.get("_id"), "domain": row.get("domain"), "sub": row.get("sub")}
    line = f"  doc {i:2d} [{str(row.get('domain'))[:22]:22s}]"
    for lab, served in models:
        n, txt = gen(served, row["doc"])
        res[lab].append(n); rec[lab + "_tok"] = n; rec[lab + "_text"] = txt
        line += f"  {lab}={n:4d}"
    out.append(rec); print(line, flush=True)
json.dump(out, open(a.out, "w"), indent=1)

print("\n=== summary length (completion tokens, greedy, neutral prompt) ===")
for lab, _ in models:
    L = res[lab]
    print(f"  {lab:12s}: mean={st.mean(L):7.1f}  median={st.median(L):6.0f}  "
          f"std={st.pstdev(L):6.1f}  min={min(L)} max={max(L)}")
