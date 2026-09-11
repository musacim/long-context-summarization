import argparse, json
ap = argparse.ArgumentParser()
ap.add_argument("--docs", required=True)
ap.add_argument("--summaries", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
docs = {json.loads(l)["_id"]: json.loads(l)["doc"] for l in open(a.docs)}
n = skipped = 0
with open(a.out, "w") as f:
    for l in open(a.summaries):
        r = json.loads(l)
        if "error" in r or not r.get("summary") or r["_id"] not in docs:
            skipped += 1; continue
        f.write(json.dumps({"_id": r["_id"], "doc": docs[r["_id"]],
                            "summary": r["summary"]}) + "\n")
        n += 1
print(f"wrote {a.out}  n={n} skipped={skipped}")
