import argparse, json, random, re, statistics as st
from math import comb

ap = argparse.ArgumentParser()
ap.add_argument("outputs")
ap.add_argument("--base", default="base")
ap.add_argument("--cap", type=int, default=3072)
ap.add_argument("--rep-threshold", type=float, default=0.30)
a = ap.parse_args()
rows = json.load(open(a.outputs))
labels = sorted({k[:-4] for k in rows[0] if k.endswith("_tok")})

def rep(t):
    w = re.findall(r"\w+", t.lower())
    if len(w) < 40: return 0.0
    g = [tuple(w[i:i+8]) for i in range(len(w) - 7)]
    return 1 - len(set(g)) / len(g)

bad = {r["i"] for r in rows for lab in labels
       if r[lab + "_tok"] >= a.cap or rep(r[lab + "_text"]) > a.rep_threshold}
clean = [r for r in rows if r["i"] not in bad]
print(f"excluded docs (capped/degenerate in any arm): {sorted(bad)}  ->  n={len(clean)}\n")

for lab in labels:
    L = sorted(r[lab + "_tok"] for r in clean)
    print(f"  {lab:12s}: low={L[0]:5d}  median={st.median(L):6.0f}  high={L[-1]:5d}")

base = [r[a.base + "_tok"] for r in clean]
print(f"\n=== paired vs {a.base} (bootstrap 95% CI; exact sign test) ===")
for lab in labels:
    if lab == a.base: continue
    d = [r[lab + "_tok"] - b for r, b in zip(clean, base)]
    n = len(d); k = sum(1 for x in d if x < 0)
    p = min(1.0, sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2**n * 2)
    rnd = random.Random(0)
    m = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n for _ in range(20000))
    direction = f"{k}/{n} shorter" if k > n - k else f"{n-k}/{n} longer"
    print(f"  {lab:12s}: delta={st.mean(d):+8.1f}  CI=[{m[500]:+.0f},{m[19500]:+.0f}]  "
          f"{direction}  p={p:.3g}")
