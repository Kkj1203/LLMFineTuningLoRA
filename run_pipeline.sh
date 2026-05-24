#!/bin/bash
# =============================================================================
# run_pipeline.sh — One-shot execution: setup + train + evaluate
# Run this on your AMD MI300X instance AFTER uploading project files.
# =============================================================================

set -e

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Project 4: LoRA SFT + DPO on AMD MI300X               ║"
echo "║   Estimated time: ~35-45 minutes | Cost: ~\$1.20-1.50    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check we're in the right directory
if [ ! -f "train_sft_dpo.py" ]; then
    echo "❌ train_sft_dpo.py not found."
    echo "   Run this script from the LLMFineTuningLoRA_P4 directory."
    exit 1
fi

if [ ! -f "data/json_extraction_dataset.jsonl" ]; then
    echo "❌ data/json_extraction_dataset.jsonl not found."
    echo "   Make sure you uploaded the full project folder."
    exit 1
fi

# ── Step 1: Setup environment ─────────────────────────────────────────────────
echo "▶ Step 1/3 — Installing dependencies..."
bash setup_amd.sh

# ── Step 2: Run full pipeline ─────────────────────────────────────────────────
echo ""
echo "▶ Step 2/3 — Running SFT + DPO training pipeline..."
echo "   Logging to training.log"
python3 train_sft_dpo.py 2>&1 | tee training.log

# ── Step 3: Run DPO evaluation report ────────────────────────────────────────
echo ""
echo "▶ Step 3/3 — Generating DPO evaluation report..."
python3 dpo_evaluate.py 2>&1 | tee -a training.log

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   ✅ Pipeline complete! Files ready to download:         ║"
echo "║                                                          ║"
echo "║   results/three_stage_metrics.json   ← main results     ║"
echo "║   results/dpo_metrics.json           ← DPO specific     ║"
echo "║   results/dpo_report.json            ← win rate etc.    ║"
echo "║   results/metrics.json               ← backward compat  ║"
echo "║   training.log                       ← full log         ║"
echo "║                                                          ║"
echo "║   ⚠️  DESTROY YOUR INSTANCE NOW to stop billing!         ║"
echo "║   Dashboard → GPU Droplets → Destroy                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
