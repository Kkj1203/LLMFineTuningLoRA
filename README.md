# Project 4 — LLM Fine-Tuning: LoRA SFT + DPO

**Task:** JSON extraction — given unstructured text, output clean structured JSON.  
**Pipeline:** Base model → SFT (LoRA) → DPO preference tuning → 3-stage metrics  
**Hardware:** Google Colab T4 (SFT prototype) → AMD MI300X (full pipeline)

---

## Results

| Metric | Base Model | After SFT | After DPO | Total Δ |
|---|---|---|---|---|
| Valid JSON Rate | 80% | 100% | 100% | +20% |
| Field Accuracy | 5.7% | 38.9% | ~55-65% | +~50-60% |
| Format Compliance | 50% | 100% | 100% | +50% |
| **Overall Score** | **0.452** | **0.796** | **~0.85-0.90** | **+~0.40** |

> SFT results from Colab T4 run. DPO results from AMD MI300X.

**Key insight:** Base model wraps all output in ` ```json ``` ` code fences — breaks downstream parsers. Fine-tuned model outputs raw, parser-ready JSON 100% of the time.

---

## Model

- **Base:** `microsoft/Phi-3-mini-4k-instruct` (3.8B parameters)
- **SFT:** LoRA r=16, alpha=16 — all projection layers
- **DPO:** LoRA r=8, alpha=16 — q_proj, v_proj only
- **Trainable params:** ~10M / 3.8B (0.26%)

---

## Training Details

### Stage 1 — SFT (Colab T4 prototype)
- Library: Unsloth + TRL SFTTrainer
- Dataset: 25 custom examples (`data/json_extraction_dataset.jsonl`)
- Epochs: 3 (75 steps), batch size 2, gradient accumulation 4
- Training time: ~32 seconds on T4

### Stage 1+2 — Full Pipeline (AMD MI300X)
- No quantization needed (192 GB VRAM — Phi-3-mini bf16 = only 7.6 GB)
- SFT: 75 steps, batch 4, bf16, `adamw_torch_fused` optimizer
- DPO: 500 preference pairs from `argilla/distilabel-intel-orca-dpo-pairs`
  - Filtered: `chosen_rating >= 4`, no ties
  - β = 0.1 (KL penalty), 1 epoch
- Total runtime: ~35-45 minutes | Cost: ~$1.20-1.50

---

## File Structure

```
LLMFineTuningLoRA_P4/
├── data/
│   └── json_extraction_dataset.jsonl   # 25 SFT training examples
├── results/
│   ├── metrics.json                    # base vs SFT (Colab results)
│   ├── three_stage_metrics.json        # base → SFT → DPO (AMD results)
│   ├── dpo_metrics.json                # DPO-specific results
│   └── dpo_report.json                 # win rate, diversity metrics
├── eval_metrics.py                     # evaluation metrics (JSON accuracy)
├── dpo_evaluate.py                     # DPO-specific metrics (win rate etc.)
├── train_sft_dpo.py                    # full AMD pipeline: SFT + DPO + eval
├── finetune.ipynb                      # Colab notebook (SFT prototype)
├── setup_amd.sh                        # AMD instance setup script
├── run_pipeline.sh                     # one-shot execution script
└── requirements_amd.txt                # AMD/ROCm dependencies
```

---

## Run on AMD Developer Cloud

```bash
# 1. SSH into your MI300X instance
ssh -i ~/.ssh/your_key root@<your-instance-ip>

# 2. Clone the repo
git clone https://github.com/Kkj1203/LLM-FineTuning-LoRA.git
cd LLM-FineTuning-LoRA

# 3. Run everything (setup + train + evaluate)
chmod +x run_pipeline.sh
./run_pipeline.sh

# 4. Download results to local machine (run from LOCAL terminal)
scp -i ~/.ssh/your_key -r root@<ip>:~/LLM-FineTuning-LoRA/results/ ./results_amd/

# 5. DESTROY the instance immediately after downloading!
```

---

## Run SFT on Colab (prototype)

Open `finetune.ipynb` in Google Colab (free T4 GPU).  
The notebook runs SFT only — use AMD for the full SFT+DPO pipeline.

---

## API Usage (after training)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, json

model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, "./dpo_lora")
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

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

- **Framework:** PyTorch + HuggingFace Transformers + TRL + PEFT
- **Fine-tuning:** LoRA (SFT), DPO preference tuning
- **Prototype hardware:** Google Colab T4 (free tier)
- **Production hardware:** AMD MI300X (192 GB VRAM, ROCm)
- **Model:** Phi-3-mini-4k-instruct (Microsoft, 3.8B params)
