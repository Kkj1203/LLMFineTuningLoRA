"""
train_sft_dpo.py — Full Pipeline: SFT LoRA → DPO on AMD MI300X
===============================================================
Stage 1: SFT fine-tuning (LoRA) on JSON extraction task
Stage 2: DPO preference tuning using argilla/distilabel-intel-orca-dpo-pairs
Stage 3: Evaluate all three checkpoints (base, SFT, DPO) and save metrics

Optimized for: AMD MI300X (192 GB VRAM) — NO 4-bit quantization needed!
Expected runtime: ~30–45 minutes total on 1x MI300X
Expected cost: ~$1.00–1.50 at $1.99/hr

Usage:
    python3 train_sft_dpo.py
    python3 train_sft_dpo.py --skip-sft   # if SFT adapter already saved
    python3 train_sft_dpo.py --dpo-only   # alias for --skip-sft
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ─── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--skip-sft",  action="store_true", help="Skip SFT, load existing ./sft_lora/")
parser.add_argument("--dpo-only",  action="store_true", help="Alias for --skip-sft")
parser.add_argument("--model",     default="microsoft/Phi-3-mini-4k-instruct", help="Base model id")
parser.add_argument("--sft-steps", type=int, default=75,  help="SFT max steps (default 75 ≈ 3 epochs on 25 examples)")
parser.add_argument("--dpo-pairs", type=int, default=500, help="Number of DPO preference pairs to train on")
parser.add_argument("--dpo-epochs",type=int, default=1,   help="DPO training epochs")
parser.add_argument("--beta",      type=float, default=0.1, help="DPO beta (KL penalty)")
args = parser.parse_args()

if args.dpo_only:
    args.skip_sft = True

# ─── Imports ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  Importing libraries...")
print("="*60)

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
from datasets import load_dataset, Dataset

print(f"PyTorch:        {torch.__version__}")
print(f"Transformers:   {transformers.__version__}")
print(f"ROCm/HIP:       {torch.version.hip if hasattr(torch.version, 'hip') else 'N/A'}")
print(f"GPU count:      {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU[{i}]: {p.name} | {p.total_memory/1024**3:.1f} GB VRAM")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Path("results").mkdir(exist_ok=True)
Path("sft_lora").mkdir(exist_ok=True)
Path("dpo_lora").mkdir(exist_ok=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
MODEL_ID  = args.model
MAX_SEQ   = 512
DTYPE     = torch.bfloat16   # MI300X supports bfloat16 natively — faster than fp16

# LoRA config for SFT
LORA_CFG = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# LoRA config for DPO (smaller rank for fine-grained preference adjustment)
DPO_LORA_CFG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print(f"\nBase model: {MODEL_ID}")
print(f"SFT LoRA:   r={LORA_CFG.r}, alpha={LORA_CFG.lora_alpha}")
print(f"DPO LoRA:   r={DPO_LORA_CFG.r}, alpha={DPO_LORA_CFG.lora_alpha}")
print(f"Precision:  bf16 (no quantization — MI300X has 192GB VRAM!)")

# ==============================================================================
# HELPER: Load base model + tokenizer
# ==============================================================================
def load_base_model(adapter_path=None):
    """Load model in bf16. If adapter_path given, load LoRA on top."""
    print(f"\nLoading tokenizer from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model (bf16, no quantization)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",   # faster; falls back to eager if not available
    )
    model.config.use_cache = False
    print(f"  Loaded in {time.time()-t0:.1f}s")

    if adapter_path and Path(adapter_path).exists():
        print(f"  Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


# ==============================================================================
# HELPER: Quick inference for evaluation
# ==============================================================================
PROMPT_TEMPLATE = """<|user|>
Extract structured JSON from the following text. Output ONLY valid JSON, nothing else.

Text: {input}
<|end|>
<|assistant|>
"""

def run_inference(model, tokenizer, test_cases, label="model", max_new=256):
    """Run TEST_CASES through model and return raw text outputs."""
    print(f"\n  Running inference [{label}] on {len(test_cases)} test cases...")
    model.eval()
    outputs = []
    for tc in test_cases:
        prompt = PROMPT_TEMPLATE.format(input=tc["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Decode only the new tokens
        new_tokens = ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        outputs.append(text)
        print(f"    [{tc['input'][:40]}...] → {text[:60]}...")
    return outputs


# ==============================================================================
# STAGE 0: Load TEST_CASES from eval_metrics.py
# ==============================================================================
print("\n" + "="*60)
print("  STAGE 0 — Loading evaluation test cases")
print("="*60)

# Import from eval_metrics.py (renamed to avoid HuggingFace evaluate library conflict)
sys.path.insert(0, str(Path(__file__).parent))
from eval_metrics import TEST_CASES, compare_and_save, evaluate_outputs

print(f"  Loaded {len(TEST_CASES)} held-out test cases from eval_metrics.py")


# ==============================================================================
# STAGE 1: Base Model Evaluation (before any training)
# ==============================================================================
print("\n" + "="*60)
print("  STAGE 1 — Base Model Evaluation")
print("="*60)

base_model, tokenizer = load_base_model()
base_outputs = run_inference(base_model, tokenizer, TEST_CASES, label="base_model")

# Free memory before SFT
del base_model
torch.cuda.empty_cache()
print("  GPU memory cleared after base eval")


# ==============================================================================
# STAGE 2: SFT Fine-Tuning (LoRA)
# ==============================================================================
print("\n" + "="*60)
print(f"  STAGE 2 — SFT Fine-Tuning {'[SKIPPED — loading saved adapter]' if args.skip_sft else ''}")
print("="*60)

if not args.skip_sft:
    # ── Load dataset ──────────────────────────────────────────────────────────
    print("  Loading SFT dataset from data/json_extraction_dataset.jsonl...")
    data_path = Path(__file__).parent / "data" / "json_extraction_dataset.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Upload data/json_extraction_dataset.jsonl to your instance."
        )

    raw_data = []
    with open(data_path) as f:
        for line in f:
            raw_data.append(json.loads(line.strip()))

    print(f"  Loaded {len(raw_data)} training examples")

    # Format as chat messages
    def format_sft_example(example):
        user_msg = f"Extract structured JSON from the following text. Output ONLY valid JSON, nothing else.\n\nText: {example['input']}"
        assistant_msg = json.dumps(example["output"]) if isinstance(example["output"], dict) else example["output"]
        return {
            "messages": [
                {"role": "user",      "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }

    sft_dataset = Dataset.from_list([format_sft_example(ex) for ex in raw_data])
    print(f"  SFT dataset formatted: {len(sft_dataset)} examples")

    # ── Load model for SFT ────────────────────────────────────────────────────
    sft_model, _ = load_base_model()
    sft_model = get_peft_model(sft_model, LORA_CFG)
    sft_model.print_trainable_parameters()

    # ── SFT Training config ───────────────────────────────────────────────────
    sft_config = SFTConfig(
        output_dir="./sft_lora",
        max_steps=args.sft_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        gradient_checkpointing=False,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        optim="adamw_torch_fused",         # fused optimizer = faster on ROCm
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        max_seq_length=MAX_SEQ,
        report_to="none",
    )

    print(f"\n  Starting SFT training ({args.sft_steps} steps)...")
    t_sft = time.time()

    trainer = SFTTrainer(
        model=sft_model,
        train_dataset=sft_dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()

    elapsed = time.time() - t_sft
    print(f"\n  ✅ SFT done in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"     Final loss: {train_result.training_loss:.4f}")

    # Save adapter
    trainer.save_model("./sft_lora")
    tokenizer.save_pretrained("./sft_lora")
    print("  Adapter saved to ./sft_lora/")

    # Free memory
    del sft_model, trainer
    torch.cuda.empty_cache()

# ── Evaluate SFT model ────────────────────────────────────────────────────────
print("\n  Evaluating SFT checkpoint...")
sft_eval_model, tokenizer = load_base_model(adapter_path="./sft_lora")
sft_outputs = run_inference(sft_eval_model, tokenizer, TEST_CASES, label="sft_model")

del sft_eval_model
torch.cuda.empty_cache()


# ==============================================================================
# STAGE 3: DPO Fine-Tuning
# ==============================================================================
print("\n" + "="*60)
print("  STAGE 3 — DPO Preference Tuning")
print("="*60)

# ── Load DPO preference dataset ───────────────────────────────────────────────
print(f"  Loading argilla/distilabel-intel-orca-dpo-pairs...")
print(f"  Filtering to {args.dpo_pairs} high-quality pairs...")

raw_dpo = load_dataset("argilla/distilabel-intel-orca-dpo-pairs", split="train")
print(f"  Raw dataset size: {len(raw_dpo)}")

# Defensive field accessor (handles multiple naming conventions)
def get_field(example, *keys, default=None):
    for k in keys:
        if k in example and example[k] is not None:
            return example[k]
    return default

def dpo_filter(example):
    """Keep only high-quality, non-tied preference pairs."""
    # Try multiple field names for rating
    chosen_score = get_field(example, "chosen_rating", "rating_chosen", "chosen_score", default=0)
    # Try multiple field names for prompt
    prompt = get_field(example, "input", "question", "prompt", "instruction", default="")
    # Check for tie
    rejected_score = get_field(example, "rejected_rating", "rating_rejected", "rejected_score", default=0)
    is_tie = (chosen_score == rejected_score) if (chosen_score and rejected_score) else False
    
    return (
        not is_tie and
        float(chosen_score) >= 4.0 and
        bool(prompt) and
        bool(example.get("chosen")) and
        bool(example.get("rejected"))
    )

dpo_filtered = raw_dpo.filter(dpo_filter)
print(f"  After quality filter: {len(dpo_filtered)} pairs")

# Sample the requested number of pairs
n = min(args.dpo_pairs, len(dpo_filtered))
dpo_data = dpo_filtered.shuffle(seed=42).select(range(n))
print(f"  Using {n} preference pairs for DPO training")

# Format: DPOTrainer expects dict with 'prompt', 'chosen', 'rejected' as plain strings
def format_dpo_example(example):
    """Normalize chosen/rejected to plain strings."""
    def to_str(val):
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            # Chat-format list of messages — extract last assistant turn
            for msg in reversed(val):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return str(val[-1]) if val else ""
        return str(val)

    # Try multiple field names
    prompt  = get_field(example, "input", "question", "prompt", "instruction", default="")
    chosen  = to_str(example.get("chosen", ""))
    rejected= to_str(example.get("rejected", ""))

    # Skip if any field is empty
    if not all([prompt, chosen, rejected]):
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

formatted = [format_dpo_example(ex) for ex in dpo_data]
formatted  = [ex for ex in formatted if ex is not None]
dpo_dataset = Dataset.from_list(formatted)
print(f"  DPO dataset ready: {len(dpo_dataset)} formatted pairs")

# ── Load SFT model as DPO policy ──────────────────────────────────────────────
print("\n  Loading SFT model as DPO policy (starting point)...")
dpo_base, tokenizer = load_base_model(adapter_path="./sft_lora")

# Merge SFT adapter into base weights (works fine in bf16, no quantization issues!)
print("  Merging SFT adapter into base weights for DPO...")
dpo_model = dpo_base.merge_and_unload()
dpo_model.config.use_cache = False

# Apply new LoRA adapter for DPO training
dpo_model = get_peft_model(dpo_model, DPO_LORA_CFG)
dpo_model.print_trainable_parameters()

# ── DPO Training config ───────────────────────────────────────────────────────
dpo_config = DPOConfig(
    output_dir="./dpo_lora",
    num_train_epochs=args.dpo_epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    learning_rate=5e-5,             # ~4x smaller than SFT
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="adamw_torch_fused",
    bf16=True,
    beta=args.beta,                 # KL penalty (0.1 = standard)
    max_length=MAX_SEQ,
    max_prompt_length=256,
    logging_steps=20,
    save_strategy="no",
    report_to="none",
    remove_unused_columns=False,
)

print(f"\n  Starting DPO training...")
print(f"  Pairs: {len(dpo_dataset)} | Epochs: {args.dpo_epochs} | β: {args.beta}")
t_dpo = time.time()

dpo_trainer = DPOTrainer(
    model=dpo_model,
    ref_model=None,        # None = use implicit reference from frozen base weights
    args=dpo_config,
    train_dataset=dpo_dataset,
    tokenizer=tokenizer,
)

dpo_result = dpo_trainer.train()
elapsed_dpo = time.time() - t_dpo

print(f"\n  ✅ DPO done in {elapsed_dpo:.0f}s ({elapsed_dpo/60:.1f} min)")
print(f"     Final DPO loss: {dpo_result.training_loss:.4f}")

dpo_trainer.save_model("./dpo_lora")
print("  DPO adapter saved to ./dpo_lora/")

# Capture DPO training stats
dpo_train_log = {
    "pairs_used":    len(dpo_dataset),
    "epochs":        args.dpo_epochs,
    "beta":          args.beta,
    "final_loss":    round(dpo_result.training_loss, 4),
    "train_time_s":  round(elapsed_dpo, 1),
    "model":         MODEL_ID,
}

del dpo_model, dpo_trainer
torch.cuda.empty_cache()

# Load DPO checkpoint for evaluation
print("\n  Evaluating DPO checkpoint...")
dpo_eval_model, _ = load_base_model(adapter_path="./dpo_lora")
dpo_outputs = run_inference(dpo_eval_model, tokenizer, TEST_CASES, label="dpo_model")

del dpo_eval_model
torch.cuda.empty_cache()


# ==============================================================================
# STAGE 4: Full 3-Stage Evaluation & Save
# ==============================================================================
print("\n" + "="*60)
print("  STAGE 4 — Three-Stage Metric Comparison")
print("="*60)

print("\n── BASE MODEL ─────────────────────────────────────────────")
base_metrics = evaluate_outputs(base_outputs,  label="base_model")

print("\n── SFT (LoRA) ─────────────────────────────────────────────")
sft_metrics  = evaluate_outputs(sft_outputs,   label="sft_model")

print("\n── DPO ────────────────────────────────────────────────────")
dpo_metrics  = evaluate_outputs(dpo_outputs,   label="dpo_model")

# ── Delta calculations ────────────────────────────────────────────────────────
sft_delta = {
    "valid_json":        round(sft_metrics["valid_json_rate"]   - base_metrics["valid_json_rate"],   3),
    "field_accuracy":    round(sft_metrics["field_accuracy"]    - base_metrics["field_accuracy"],    3),
    "format_compliance": round(sft_metrics["format_compliance"] - base_metrics["format_compliance"], 3),
    "overall":           round(sft_metrics["overall_score"]     - base_metrics["overall_score"],     3),
}
dpo_delta = {
    "valid_json":        round(dpo_metrics["valid_json_rate"]   - sft_metrics["valid_json_rate"],    3),
    "field_accuracy":    round(dpo_metrics["field_accuracy"]    - sft_metrics["field_accuracy"],     3),
    "format_compliance": round(dpo_metrics["format_compliance"] - sft_metrics["format_compliance"],  3),
    "overall":           round(dpo_metrics["overall_score"]     - sft_metrics["overall_score"],      3),
}
total_delta = {
    "valid_json":        round(dpo_metrics["valid_json_rate"]   - base_metrics["valid_json_rate"],   3),
    "field_accuracy":    round(dpo_metrics["field_accuracy"]    - base_metrics["field_accuracy"],    3),
    "format_compliance": round(dpo_metrics["format_compliance"] - base_metrics["format_compliance"], 3),
    "overall":           round(dpo_metrics["overall_score"]     - base_metrics["overall_score"],     3),
}

# ── Pretty print summary ──────────────────────────────────────────────────────
print("\n" + "="*70)
print("  THREE-STAGE COMPARISON SUMMARY")
print("="*70)
fmt = "{:<22} {:>10} {:>12} {:>10} {:>12} {:>12}"
print(fmt.format("Metric", "Base", "SFT (Δ)", "DPO", "DPO Δ SFT", "Total Δ"))
print("-"*70)

def pct(v): return f"{v:.1%}"
def d(v):   return f"+{v:.1%}" if v >= 0 else f"{v:.1%}"

print(fmt.format(
    "Valid JSON Rate",
    pct(base_metrics["valid_json_rate"]),
    f"{pct(sft_metrics['valid_json_rate'])} ({d(sft_delta['valid_json'])})",
    pct(dpo_metrics["valid_json_rate"]),
    d(dpo_delta["valid_json"]),
    d(total_delta["valid_json"]),
))
print(fmt.format(
    "Field Accuracy",
    pct(base_metrics["field_accuracy"]),
    f"{pct(sft_metrics['field_accuracy'])} ({d(sft_delta['field_accuracy'])})",
    pct(dpo_metrics["field_accuracy"]),
    d(dpo_delta["field_accuracy"]),
    d(total_delta["field_accuracy"]),
))
print(fmt.format(
    "Format Compliance",
    pct(base_metrics["format_compliance"]),
    f"{pct(sft_metrics['format_compliance'])} ({d(sft_delta['format_compliance'])})",
    pct(dpo_metrics["format_compliance"]),
    d(dpo_delta["format_compliance"]),
    d(total_delta["format_compliance"]),
))
print(fmt.format(
    "Overall Score",
    f"{base_metrics['overall_score']:.3f}",
    f"{sft_metrics['overall_score']:.3f} ({d(sft_delta['overall'])})",
    f"{dpo_metrics['overall_score']:.3f}",
    d(dpo_delta["overall"]),
    d(total_delta["overall"]),
))
print("="*70)

# ── Save full results ─────────────────────────────────────────────────────────
full_results = {
    "base_model":   base_metrics,
    "sft_model":    sft_metrics,
    "dpo_model":    dpo_metrics,
    "sft_vs_base":  sft_delta,
    "dpo_vs_sft":   dpo_delta,
    "dpo_vs_base":  total_delta,
    "dpo_training": dpo_train_log,
    "hardware":     {
        "gpu":   torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else "unknown",
        "vram_gb": torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else 0,
    },
}

results_path = Path("results") / "three_stage_metrics.json"
with open(results_path, "w") as f:
    json.dump(full_results, f, indent=2)

print(f"\n💾 Full results saved → {results_path}")

# Also save backward-compatible metrics.json (base vs SFT for existing README)
compare_and_save(base_outputs, sft_outputs)

# ── Save DPO-only metrics ─────────────────────────────────────────────────────
dpo_results_path = Path("results") / "dpo_metrics.json"
with open(dpo_results_path, "w") as f:
    json.dump({
        "before_dpo": sft_metrics,
        "after_dpo":  dpo_metrics,
        "improvement": dpo_delta,
        "training_config": dpo_train_log,
    }, f, indent=2)

print(f"💾 DPO-specific results → {dpo_results_path}")

print("\n" + "="*60)
print("  ✅ PIPELINE COMPLETE!")
print("="*60)
print("\nFiles to commit to GitHub:")
print("  results/three_stage_metrics.json  ← main results")
print("  results/dpo_metrics.json          ← DPO-specific")
print("  results/metrics.json              ← backward compat")
print("  sft_lora/                         ← SFT adapter (optional)")
print("  dpo_lora/                         ← DPO adapter (optional)")
print("\n⚠️  IMPORTANT: DESTROY your AMD instance now to stop billing!")
print("   Dashboard → GPU Droplets → [your instance] → Destroy")
