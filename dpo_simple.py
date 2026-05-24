"""
dpo_simple.py — Custom DPO training loop (no TRL dependency)
Loads SFT adapter, trains DPO using raw PyTorch, saves adapter.
"""
import json, time, torch
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel
from datasets import load_dataset

Path("results").mkdir(exist_ok=True)
Path("dpo_lora").mkdir(exist_ok=True)

DEVICE   = "cuda"
DTYPE    = torch.bfloat16
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
BETA     = 0.1
LR       = 5e-5
EPOCHS   = 1
MAX_LEN  = 384
PAIRS    = 500
BATCH    = 2

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading SFT model as DPO policy...")
policy = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto",
    trust_remote_code=True, attn_implementation="eager"
)
policy = PeftModel.from_pretrained(policy, "./sft_lora")
policy = policy.merge_and_unload()

dpo_lora = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
)
policy = get_peft_model(policy, dpo_lora)
policy.print_trainable_parameters()

print("Loading frozen reference model...")
reference = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=DTYPE, device_map="auto",
    trust_remote_code=True, attn_implementation="eager"
)
reference = PeftModel.from_pretrained(reference, "./sft_lora")
reference = reference.merge_and_unload()
reference.eval()
for p in reference.parameters():
    p.requires_grad = False

print("Loading DPO dataset...")
ds = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
filtered = ds.filter(lambda x: (x.get("chosen_score") or 0) >= 4.0
                     and x.get("input") and x.get("chosen") and x.get("rejected"))
filtered = filtered.shuffle(seed=42).select(range(min(PAIRS, len(filtered))))
print(f"Using {len(filtered)} pairs")

def tokenize_pair(example):
    def enc(text):
        return tokenizer(text, max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
    prompt   = example["input"]
    chosen   = enc(prompt + " " + example["chosen"])
    rejected = enc(prompt + " " + example["rejected"])
    return {
        "chosen_ids":   chosen["input_ids"][0],
        "chosen_mask":  chosen["attention_mask"][0],
        "rejected_ids": rejected["input_ids"][0],
        "rejected_mask":rejected["attention_mask"][0],
    }

tokenized = [tokenize_pair(ex) for ex in filtered]

def collate(batch):
    return {
        "chosen_ids":   torch.stack([b["chosen_ids"]   for b in batch]),
        "chosen_mask":  torch.stack([b["chosen_mask"]  for b in batch]),
        "rejected_ids": torch.stack([b["rejected_ids"] for b in batch]),
        "rejected_mask":torch.stack([b["rejected_mask"]for b in batch]),
    }

loader = DataLoader(tokenized, batch_size=BATCH, shuffle=True, collate_fn=collate)

def log_probs(model, ids, mask):
    with torch.no_grad() if not model.training else torch.enable_grad():
        out      = model(input_ids=ids, attention_mask=mask)
        logits   = out.logits[:, :-1]
        labels   = ids[:, 1:]
        lp       = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = lp.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        return (token_lp * mask[:, 1:]).sum(-1)

def dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    pi_logratios  = policy_chosen  - policy_rejected
    ref_logratios = ref_chosen     - ref_rejected
    return -torch.nn.functional.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()

optimizer = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01)
policy.train()

print(f"\nStarting DPO training | {len(loader)} steps | {EPOCHS} epoch(s)")
t0 = time.time()
total_loss = 0.0
steps = 0

for epoch in range(EPOCHS):
    for batch in loader:
        cids  = batch["chosen_ids"].to(DEVICE)
        cmask = batch["chosen_mask"].to(DEVICE)
        rids  = batch["rejected_ids"].to(DEVICE)
        rmask = batch["rejected_mask"].to(DEVICE)

        pol_c = log_probs(policy,    cids, cmask)
        pol_r = log_probs(policy,    rids, rmask)
        with torch.no_grad():
            ref_c = log_probs(reference, cids, cmask)
            ref_r = log_probs(reference, rids, rmask)

        loss = dpo_loss(pol_c, pol_r, ref_c, ref_r, BETA)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        steps      += 1
        if steps % 20 == 0:
            print(f"  Step {steps}/{len(loader)*EPOCHS} | loss: {total_loss/steps:.4f}")

elapsed    = time.time() - t0
final_loss = total_loss / steps
print(f"\n✅ DPO done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"   Final avg loss: {final_loss:.4f}")

policy.save_pretrained("./dpo_lora")
tokenizer.save_pretrained("./dpo_lora")
print("Saved → ./dpo_lora/")

with open("results/dpo_metrics.json", "w") as f:
    json.dump({
        "training_config": {"pairs": PAIRS, "epochs": EPOCHS, "beta": BETA, "lr": LR},
        "final_loss":      round(final_loss, 4),
        "train_time_s":    round(elapsed, 1),
        "model":           MODEL_ID,
        "hardware":        "AMD MI300X 192GB",
    }, f, indent=2)

print("Results saved → results/dpo_metrics.json")
