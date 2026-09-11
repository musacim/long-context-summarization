import os, json, argparse, torch
from dataclasses import dataclass
from transformers import (AutoTokenizer, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model

NEUTRAL_INSTR = "Summarize the following document.\n\n"
MAX_LEN = int(os.environ.get('MAX_LEN', 33000))
DOC_BUDGET = 32200

def build_example(tok, doc, summary):
    dids = tok(doc, add_special_tokens=False).input_ids[:DOC_BUDGET]
    doc = tok.decode(dids)
    user = [{"role": "user", "content": NEUTRAL_INSTR + doc}]
    full = user + [{"role": "assistant", "content": summary}]
    ptext = tok.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
    ftext = tok.apply_chat_template(full, tokenize=False, add_generation_prompt=False)
    pids = tok(ptext, add_special_tokens=False).input_ids
    fids = tok(ftext, add_special_tokens=False).input_ids
    assert fids[:len(pids)] == pids, "chat-template prefix mismatch -> mask would be wrong"
    truncated = len(fids) > MAX_LEN
    fids = fids[:MAX_LEN]
    build_example.n_trunc = getattr(build_example, "n_trunc", 0) + (1 if truncated else 0)
    labels = [-100] * len(pids) + fids[len(pids):]
    labels = labels[:len(fids)]
    return {"input_ids": fids, "labels": labels}

@dataclass
class Collator:
    pad_id: int
    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        ids, lab, att = [], [], []
        for f in feats:
            n = m - len(f["input_ids"])
            ids.append(f["input_ids"] + [self.pad_id] * n)
            lab.append(f["labels"] + [-100] * n)
            att.append([1] * len(f["input_ids"]) + [0] * n)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab),
                "attention_mask": torch.tensor(att)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    from liger_kernel.transformers import AutoLigerKernelForCausalLM
    model = AutoLigerKernelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))})
    model.config.use_cache = False
    model.enable_input_require_grads()

    lora = LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05,
                      task_type="CAUSAL_LM", target_modules=[
                          "q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    if int(os.environ.get("RANK", 0)) == 0:
        model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.data)]
    ds = [build_example(tok, r["doc"], r["summary"]) for r in rows]
    if int(os.environ.get("RANK", 0)) == 0:
        import statistics as st
        L = [sum(1 for x in e["labels"] if x != -100) for e in ds]
        print(f"[data] {len(ds)} ex | completion-tok mean={st.mean(L):.0f} "
              f"seqlen max={max(len(e['input_ids']) for e in ds)} "
              f"| MAX_LEN={MAX_LEN} TRUNCATED={getattr(build_example,'n_trunc',0)}", flush=True)
        assert getattr(build_example,'n_trunc',0) == 0, "examples were truncated -- raise MAX_LEN"

    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr,
        per_device_train_batch_size=1, gradient_accumulation_steps=2,
        lr_scheduler_type="cosine", warmup_steps=1, logging_steps=1,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="epoch", save_total_limit=1, report_to=[],
        remove_unused_columns=False)
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=Collator(tok.pad_token_id)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    if int(os.environ.get("RANK", 0)) == 0:
        print("SAVED adapter ->", a.out, flush=True)

if __name__ == "__main__":
    main()
