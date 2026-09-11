import argparse, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
INTOK = 32_000
N_TRAIN = 100

IBENCH_SHA = "4e4279cc06c07778f646cda1335755cf5ccdcc38fe9c8a62f1631642fc031027"
LBV2_SHA   = "15d61c22d92c96900b3c4948b6aeea218d3214b676a65df48e7b8555604c7fe2"

HELDOUT_IDS = [
    "66fcffd9bb02136c067c94c5", "66f36490821e116aacb2cc22", "66f94f9ebb02136c067c4fde",
    "6723a1ccbb02136c067d70b3", "670aac92bb02136c067d218a", "671b3cabbb02136c067d5252",
    "66fa208bbb02136c067c5fc1", "66ec0c4c821e116aacb1994a", "66f920d8bb02136c067c4b81",
    "66ec56dd821e116aacb1cd0e", "6725dc28bb02136c067d8555", "66ebd55f5a08c7b9b35e0698",
    "670cf1c0bb02136c067d26e5", "66ec3fa7821e116aacb1c75d", "66ed875e821e116aacb2023e",
    "66f8c9febb02136c067c4511", "67239fd9bb02136c067d6ff7", "66f61d7dbb02136c067c1802",
    "66f3f46f821e116aacb2ff5d", "66f599ef821e116aacb34099",
]

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()

def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def build_train(tok):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("xinrongzhang2022/InfiniteBench", "longbook_sum_eng.jsonl",
                        repo_type="dataset")
    got = sha256(p)
    if got != IBENCH_SHA:
        sys.exit(f"InfiniteBench checksum mismatch: {got} (upstream changed?)")
    seen, out = set(), []
    for line in open(p):
        c = json.loads(line)["context"]
        k = c[:2000]
        if k in seen:
            continue
        seen.add(k)
        ids = tok.encode(c).ids
        if len(ids) < INTOK:
            continue
        out.append({"_id": f"ibench_{len(out):03d}", "doc": tok.decode(ids[:INTOK]),
                    "n_tok": INTOK})
        if len(out) >= N_TRAIN:
            break
    path = os.path.join(OUT, "train_docs.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}  n={len(out)}")

def build_heldout(tok):
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("zai-org/LongBench-v2", "data.json", repo_type="dataset")
    got = sha256(p)
    if got != LBV2_SHA:
        sys.exit(f"LongBench-v2 checksum mismatch: {got} (upstream changed?)")
    rows = {r["_id"]: r for r in json.load(open(p))}
    out = []
    for _id in HELDOUT_IDS:
        r = rows[_id]
        ids = tok.encode(r["context"][:400_000]).ids
        assert len(ids) >= INTOK, _id
        out.append({"_id": _id, "doc": tok.decode(ids[:INTOK]), "n_tok": INTOK,
                    "domain": r.get("domain"), "sub": r.get("sub_domain")})
    path = os.path.join(OUT, "held_out.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {path}  n={len(out)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["train", "heldout", "t1qa"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tok = tokenizer()
    if a.only in (None, "train"):
        build_train(tok)
    if a.only in (None, "heldout"):
        build_heldout(tok)
    if a.only in (None, "t1qa"):
        build_t1_qa(tok)

def build_t1_qa(tok):
    from huggingface_hub import hf_hub_download
    man = json.load(open(os.path.join(HERE, "t1_manifest.json")))
    p = hf_hub_download("zai-org/LongBench-v2", "data.json", repo_type="dataset")
    if sha256(p) != LBV2_SHA:
        sys.exit("LongBench-v2 checksum mismatch")
    rows = {r["_id"]: r for r in json.load(open(p))}
    path = os.path.join(OUT, "t1_qa_docs.jsonl")
    with open(path, "w") as f:
        for _id in man["qa_longbench_v2_ids"]:
            r = rows[_id]
            ids = tok.encode(r["context"][:700_000]).ids
            f.write(json.dumps({"_id": _id, "sub": r.get("sub_domain"),
                                "doc": tok.decode(ids[:100_000]), "n_tok": 100_000}) + "\n")
    print(f"wrote {path}")
