# data/

- `build_docs.py` — reconstructs every document set from its source dataset
  (sha256-pinned): `--only train` (100 InfiniteBench books, 32k-token prefixes),
  `--only heldout` (20 LongBench-v2 docs by id), `--only t1qa` (10 LongBench-v2
  single-doc-QA docs, 100k-token prefixes). Outputs land in `out/` (git-ignored).
- `t1_manifest.json` — source ids for the Table-1 pilot docs: the 10 LongBench-v2
  QA ids (text rebuilt by `build_docs.py`, not shipped) and provenance for the 10
  agentic trajectories (trajectory id, SWE instance, repo, source parquet).
- `t1_agentic_docs.jsonl` — the 10 agentic docs as consumed by the experiment:
  OpenHands trajectories (nvidia/Open-SWE-Traces, CC-BY-4.0) rendered to a
  `[role]/content` message stream and truncated to a 100k-token prefix. The
  structured originals remain on Hugging Face at the manifest's `source_file`.

All token counts use the Qwen2.5-7B-Instruct tokenizer; shorter input bins are
literal prefixes of these documents.
