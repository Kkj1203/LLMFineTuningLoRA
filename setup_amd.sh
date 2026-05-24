#!/bin/bash
# =============================================================================
# setup_amd.sh — Environment Setup for AMD Developer Cloud (MI300X / ROCm)
# Run this FIRST before any training scripts.
# =============================================================================

set -e
echo "=================================================================="
echo "  AMD MI300X Environment Setup — Project 4 (LoRA + DPO)"
echo "=================================================================="

# ── 1. Verify ROCm / GPU ──────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking ROCm environment..."
rocm-smi || echo "rocm-smi not available (may be inside Docker container)"
python3 -c "import torch; print(f'PyTorch {torch.__version__} | ROCm/HIP: {getattr(torch.version, \"hip\", \"N/A\")} | GPU available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not yet installed"

# ── 2. Upgrade pip ────────────────────────────────────────────────────────────
echo ""
echo "[2/5] Upgrading pip..."
pip install --upgrade pip -q

# ── 3. Core ML libraries ──────────────────────────────────────────────────────
echo ""
echo "[3/5] Installing core training libraries..."
pip install -q \
    transformers>=4.40.0 \
    datasets>=2.18.0 \
    accelerate>=0.28.0 \
    peft>=0.10.0 \
    "trl>=0.8.6" \
    sentencepiece \
    protobuf \
    scipy \
    einops \
    huggingface_hub

# ── 4. Flash Attention (optional, speeds up training) ─────────────────────────
echo ""
echo "[4/5] Attempting Flash Attention install (ROCm)..."
pip install -q flash-attn --no-build-isolation 2>/dev/null || \
    echo "Flash attention not available — will use standard attention (still fine)"

# ── 5. Verification ───────────────────────────────────────────────────────────
echo ""
echo "[5/5] Verifying all imports..."
python3 << 'PYCHECK'
import torch, transformers, datasets, peft, trl, accelerate
print('✅ torch       :', torch.__version__)
print('✅ transformers:', transformers.__version__)
print('✅ datasets    :', datasets.__version__)
print('✅ peft        :', peft.__version__)
print('✅ trl         :', trl.__version__)
print('✅ accelerate  :', accelerate.__version__)

# GPU check
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram  = props.total_memory / 1024**3
        print(f'✅ GPU [{i}]: {props.name} | VRAM: {vram:.1f} GB')
else:
    print('⚠️  torch.cuda.is_available() = False')
    print('   This is expected on ROCm — check that HIP is working')

# Test flash attention
try:
    import flash_attn
    print('✅ flash_attn  : installed')
except ImportError:
    print('⚠️  flash_attn  : not available (will use standard attention)')
PYCHECK

echo ""
echo "=================================================================="
echo "  Setup complete! Run: python3 train_sft_dpo.py"
echo "=================================================================="
