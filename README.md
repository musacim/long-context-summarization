# Why LLMs Refuse to Summarize More in Long Contexts

Code and data for the paper. Three experiments:

| dir | experiment | paper table |
|-----|------------|-------------|
| `table1_length_curve/`      | Summary length does not scale with input (nested document chains, 1k-100k tokens) | Table 1 |
| `table2_instruction_sweep/` | Asking for more does not produce more (length instructions in tokens/words/chars) | Table 2 |
| `table3_finetune/`          | Fine-tuning moves the ceiling in both directions (LoRA on short/medium/long targets) | Table 3 |

## Obtaining the data

All document sets are reconstructed from their source datasets by
`data/build_docs.py` (sources pinned by sha256); generated artifacts are shipped
directly.

```bash
python data/build_docs.py --only train     # 100 fine-tuning docs: InfiniteBench En.Sum books, 32k-token prefixes
python data/build_docs.py --only heldout   # 20 held-out eval docs: LongBench-v2, selected by id
python data/build_docs.py --only t1qa      # 10 Table-1 QA docs: LongBench-v2 single-doc QA >=100k tokens
```

- `data/t1_agentic_docs.jsonl` — 10 real OpenHands trajectories (nvidia/Open-SWE-Traces,
  CC-BY-4.0), shipped as rendered 100k-token message streams; ids in `data/t1_manifest.json`.
- `table3_finetune/results/` — the short/medium/long supervision summaries
  (teacher-generated) and the raw per-document eval outputs behind Table 3.
