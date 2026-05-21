# 🔬 LLM Fine-Tuning with QLoRA — JSON Extraction

Fine-tuning `Phi-3-mini-4k-instruct` (3.8B params) for structured JSON extraction using **QLoRA** on a T4 GPU. Demonstrates measurable before/after improvement on a real task startups care about: turning unstructured text into structured data.

---

## 🎯 Task: Structured JSON Extraction

**Input:** Raw unstructured text  
**Output:** Clean, structured JSON

**Before fine-tuning:**
```
Input:  "John Smith ordered 3 laptops at $999 each. Ship to 123 Main St."
Output: "Sure! Here is the information extracted from the text: The customer 
         John Smith has placed an order..."   ← verbose, no JSON
```

**After fine-tuning:**
```
Input:  "John Smith ordered 3 laptops at $999 each. Ship to 123 Main St."
Output: {"customer_name": "John Smith", "item": "laptop", "quantity": 3,
         "unit_price": 999, "shipping_address": "123 Main St"}   ← clean JSON
```

---

## 📊 Results

| Metric | Base Model | Fine-Tuned (QLoRA) | Improvement |
|---|---|---|---|
| Valid JSON Rate | 80% | 100% | +20% |
| Field Accuracy | 5.7% | 38.9% | +33.2% |
| Format Compliance | 50% | 100% | +50% |
| **Overall Score** | **0.452** | **0.796** | **+0.344 (+76% relative)** |

> Trained on 25 examples × 3 epochs in **32 seconds** on T4 GPU.
> Most significant gain: format compliance 50%→100% — base model wrapped
> all output in markdown code fences, breaking downstream JSON parsers.
> Fine-tuned model outputs clean, parser-ready JSON every time.
> *Results populated after training run. See `results/metrics.json`.*

---

## 🏗️ Architecture

```
Phi-3-mini-4k-instruct (3.8B params, 4-bit quantized)
     │
     ▼
QLoRA Adapters (r=16, alpha=16)
  Target modules: q_proj, k_proj, v_proj, o_proj,
                  gate_proj, up_proj, down_proj
     │
     ▼
Trainable params: ~10M / 3.8B (0.26%)
VRAM required:    ~6GB (fits on free T4)
     │
     ▼
SFT Training: 25 examples × 3 epochs
Loss: tracked per 5 steps
     │
     ▼
LoRA adapter saved separately (~80MB)
Base model unchanged
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Base Model | `unsloth/Phi-3-mini-4k-instruct` (Microsoft) |
| Fine-Tuning Method | QLoRA (4-bit quantized LoRA) |
| Training Library | Unsloth + TRL SFTTrainer |
| Quantization | BitsAndBytes 4-bit |
| Hardware | NVIDIA T4 (Google Colab free tier) |
| Evaluation | Custom JSON accuracy metrics |

---

## 🚀 How to Reproduce

### Step 1 — Open Colab
[colab.research.google.com](https://colab.research.google.com) → New notebook → Runtime → T4 GPU

### Step 2 — Run cells in order
Paste each cell from `finetune_cells.txt` sequentially.

### Step 3 — Upload files when prompted
- Cell 5: upload `data/json_extraction_dataset.jsonl`
- Cell 6: upload `evaluate.py`

### Step 4 — Download results
Cell 9 auto-downloads `metrics.json`. Commit to this repo.

**Total time on T4:** ~25-35 minutes

---

## 📁 Project Structure

```
LLM-FineTuning-LoRA/
├── finetune_cells.txt      ← Colab notebook cells (paste in order)
├── evaluate.py             ← Before/after evaluation metrics
├── data/
│   └── json_extraction_dataset.jsonl  ← 25 training examples
├── results/
│   └── metrics.json        ← Before/after scores (auto-generated)
└── README.md
```

---

## 💡 Why JSON Extraction?

JSON extraction is one of the most commercially valuable fine-tuning tasks:

- **Document processing** — extract order/invoice data from PDFs
- **API integration** — convert natural language to structured API calls
- **Data pipelines** — parse unstructured logs into queryable records
- **Tool calling** — backbone of LLM agents

Base models are verbose and unreliable at this. A fine-tuned model produces consistent, parseable output — making it production-safe.

---

## 🗺️ Roadmap

- [x] SFT fine-tuning with QLoRA (T4 Colab)
- [ ] DPO preference tuning (AMD cloud — larger run)
- [ ] Merge LoRA adapter into base model
- [ ] Benchmark against GPT-4o-mini on same task
- [ ] Deploy fine-tuned model via Ollama

---

## 👤 Author

Built by [Keerthikrishna Jog](https://github.com/Kkj1203) as part of an AI Engineering portfolio.