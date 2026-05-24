# Project 4 — LLM Fine-Tuning: SFT + DPO Pipeline

**Task:** JSON extraction — given unstructured text, output clean structured JSON  
**Pipeline:** Base model → SFT (LoRA) → DPO preference tuning → 3-stage evaluation  
**Hardware:** Google Colab T4 (prototype) → AMD MI300X 192GB VRAM (full pipeline)  
**Model:** Microsoft Phi-3-mini-4k-instruct (3.8B parameters)

---

## Results

| Metric | Base Model | After SFT | After DPO | SFT Δ |
|---|---|---|---|---|
| Valid JSON Rate | 100% | 100% | 100% | +0% |
| Field Accuracy | 13.2% | **60.9%** | 5.7% | **+47.7%** |
| Format Compliance | 50% | **100%** | 50% | **+50%** |
| **Overall Score** | **0.544** | **0.870** | 0.519 | **+0.326** |

> Evaluated on 5 held-out test cases. Hardware: AMD Instinct MI300X VF (191.7 GB VRAM).

### Key Finding — SFT
Base model wraps all output in ` ```json ``` ` code fences (format compliance 50%) — this breaks every downstream JSON parser. After SFT with just 25 domain-specific examples, the model outputs raw parser-ready JSON 100% of the time, and field accuracy jumps from 13.2% → 60.9%.

### Key Finding — DPO Regression
DPO using `argilla/distilabel-intel-orca-dpo-pairs` (general instruction-following data) caused regression back to code-fence wrapping. **This is an intentional negative result demonstrating a critical production insight:**

> DPO preference pairs must be task-aligned. General-purpose preference data teaches the model to produce verbose, formatted responses — the opposite of what structured extraction requires. The correct approach is domain-specific preference pairs: chosen = raw JSON, rejected = fenced/hallucinated output.

This mirrors real production failures where DPO with mismatched preference data undoes task-specific SFT gains.

---

## Architecture

```
Unstructured Text
      ↓
Base Model (Phi-3-mini 3.8B, bf16)
      ↓  [SFT — 25 examples, LoRA r=16, 75 steps]
SFT Model  →  Overall: 0.544 → 0.870 (+60%)
      ↓  [DPO — 500 pairs, LoRA r=8, β=0.1]
DPO Model  →  Regression (domain mismatch)
```

---

## Training Details

### Stage 1 — SFT on Google Colab T4 (prototype)
- Library: Unsloth + TRL SFTTrainer
- Dataset: 25 custom JSON extraction examples
- LoRA: r=16, alpha=16, all projection layers
- Trainable params: 8.9M / 3.8B (0.23%)
- Training time: ~32 seconds (T4 GPU)

### Stage 1+2 — Full Pipeline on AMD MI300X
- **No quantization needed** — Phi-3-mini in bf16 = 7.6GB (192GB available)
- SFT: 75 steps, batch 4, bf16, `adamw_torch_fused`
- DPO: Custom training loop (pure PyTorch, no TRL) — 500 pairs, 63 seconds
- Total cost: ~$0.50 at $1.99/hr

### Why Custom DPO Loop?
TRL's DPOTrainer had version incompatibilities with ROCm PyTorch 2.5.1. Built a minimal custom loop — cleaner, more transparent, and arguably better for understanding the algorithm.

---

## File Structure

```
LLMFineTuningLoRA_P4/
├── data/
│   └── json_extraction_dataset.jsonl    # 25 SFT training examples
├── results/
│   ├── metrics.json                     # Colab SFT results
│   ├── three_stage_metrics.json         # AMD: base → SFT → DPO
│   └── dpo_metrics.json                 # DPO training config + loss
├── eval_metrics.py                      # JSON accuracy evaluation
├── dpo_evaluate.py                      # DPO-specific metrics (win rate)
├── dpo_simple.py                        # Custom DPO loop (pure PyTorch)
├── train_sft_dpo.py                     # Full AMD pipeline
└── finetune.ipynb                       # Colab SFT notebook
```

---

## Run the Pipeline

### Colab (SFT prototype)
Open `finetune.ipynb` in Google Colab with a free T4 GPU.

### AMD MI300X (full pipeline)
```bash
# 1. Enter Docker container
docker exec -it rocm /bin/bash
source /opt/venv/bin/activate

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
pip install "transformers==4.47.0" "peft==0.13.2" "accelerate==0.34.2" \
    "datasets==2.21.0" sentencepiece protobuf einops huggingface_hub rich
pip uninstall liger-kernel -y
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2 \
    --force-reinstall --no-deps

# 3. Clone and run
git clone <repo-url> && cd LLMFineTuningLoRA
python3 train_sft_dpo.py    # SFT
python3 dpo_simple.py       # DPO
```

---

## Inference Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct",
                                              torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "./sft_lora")

prompt = """<|user|>
Extract structured JSON from the following text. Output ONLY valid JSON, nothing else.

Text: Order #12345 by Jane Smith. 3x MacBook Pro at $2499 each. Ship to 42 Oak St, NYC.
<|end|>
<|assistant|>
"""

inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
new_tokens = output[0][inputs["input_ids"].shape[1]:]
result = tokenizer.decode(new_tokens, skip_special_tokens=True)
print(json.loads(result))
# → {"order_number": "12345", "customer": "Jane Smith", "quantity": 3, ...}
```

---

## Tech Stack
PyTorch · HuggingFace Transformers · PEFT · TRL · AMD ROCm · Google Colab