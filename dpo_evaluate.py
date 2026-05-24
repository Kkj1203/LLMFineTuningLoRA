"""
dpo_evaluate.py — DPO-Specific Metrics
======================================
Computes preference-aware metrics that go beyond JSON accuracy:

  1. Win Rate        — % of test cases where DPO output beats SFT output
                       (uses field_accuracy as a heuristic judge)
  2. Response Length — avg token count (shorter = more focused)
  3. Diversity       — unique bigram ratio (more diverse = less repetitive)

Run after train_sft_dpo.py has saved results/three_stage_metrics.json
"""

import json
from pathlib import Path
from typing import List, Dict, Any

# FIX: import from eval_metrics.py NOT evaluate (avoids HuggingFace library conflict)
from eval_metrics import TEST_CASES, parse_json_safe, field_accuracy, format_compliance


# ─── METRIC FUNCTIONS ─────────────────────────────────────────────────────────

def response_length(text: str) -> int:
    """Approximate token count (word-based heuristic)."""
    return len(text.split())


def ngram_diversity(text: str, n: int = 2) -> float:
    """
    Unique n-grams / total n-grams.
    1.0 = every n-gram is unique (max diversity).
    Near 0 = highly repetitive output.
    """
    tokens = text.lower().split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    return round(len(set(ngrams)) / len(ngrams), 3)


def heuristic_win_rate(sft_outputs: List[str], dpo_outputs: List[str]) -> Dict[str, Any]:
    """
    Compare SFT vs DPO on TEST_CASES using field_accuracy as the judge.
    Returns win/tie/loss counts and win rate.
    """
    wins = ties = losses = 0
    per_case = []

    for i, tc in enumerate(TEST_CASES):
        sft_parsed = parse_json_safe(sft_outputs[i])
        dpo_parsed = parse_json_safe(dpo_outputs[i])

        sft_acc = field_accuracy(sft_parsed, tc["expected"])
        dpo_acc = field_accuracy(dpo_parsed, tc["expected"])

        if dpo_acc > sft_acc + 0.05:
            result = "win"
            wins += 1
        elif dpo_acc < sft_acc - 0.05:
            result = "loss"
            losses += 1
        else:
            result = "tie"
            ties += 1

        per_case.append({
            "test_id":   i + 1,
            "sft_acc":   sft_acc,
            "dpo_acc":   dpo_acc,
            "result":    result,
        })

    total = len(TEST_CASES)
    win_rate = round(wins / total, 3)

    print(f"\n  Win Rate (DPO vs SFT on JSON task):")
    print(f"    Wins:   {wins}/{total}  ({wins/total:.0%})")
    print(f"    Ties:   {ties}/{total}  ({ties/total:.0%})")
    print(f"    Losses: {losses}/{total} ({losses/total:.0%})")

    return {
        "win_rate": win_rate,
        "wins":     wins,
        "ties":     ties,
        "losses":   losses,
        "per_case": per_case,
    }


def output_quality_metrics(outputs: List[str], label: str) -> Dict[str, Any]:
    """Compute length and diversity metrics over a set of outputs."""
    lengths     = [response_length(o)    for o in outputs]
    diversities = [ngram_diversity(o, 2) for o in outputs]

    return {
        "label":             label,
        "avg_length_tokens": round(sum(lengths)     / len(lengths),     1),
        "avg_diversity_2g":  round(sum(diversities) / len(diversities), 3),
        "min_length":        min(lengths),
        "max_length":        max(lengths),
    }


# ─── MAIN REPORT ──────────────────────────────────────────────────────────────

def generate_dpo_report(sft_outputs: List[str], dpo_outputs: List[str]) -> Dict[str, Any]:
    """Generate the full DPO evaluation report."""
    print("\n" + "="*55)
    print("  DPO EVALUATION REPORT")
    print("="*55)

    win_stats   = heuristic_win_rate(sft_outputs, dpo_outputs)
    sft_quality = output_quality_metrics(sft_outputs, "sft_model")
    dpo_quality = output_quality_metrics(dpo_outputs, "dpo_model")

    print(f"\n  Response Length (avg tokens):")
    print(f"    SFT: {sft_quality['avg_length_tokens']:.1f}")
    print(f"    DPO: {dpo_quality['avg_length_tokens']:.1f}  "
          f"({'shorter ↓' if dpo_quality['avg_length_tokens'] < sft_quality['avg_length_tokens'] else 'longer ↑'})")

    print(f"\n  Bigram Diversity (higher = less repetitive):")
    print(f"    SFT: {sft_quality['avg_diversity_2g']:.3f}")
    print(f"    DPO: {dpo_quality['avg_diversity_2g']:.3f}")

    report = {
        "win_rate_vs_sft": win_stats,
        "output_quality":  {"sft": sft_quality, "dpo": dpo_quality},
        "summary": {
            "dpo_win_rate":   win_stats["win_rate"],
            "dpo_avg_length": dpo_quality["avg_length_tokens"],
            "sft_avg_length": sft_quality["avg_length_tokens"],
            "dpo_diversity":  dpo_quality["avg_diversity_2g"],
            "sft_diversity":  sft_quality["avg_diversity_2g"],
        }
    }

    Path("results").mkdir(exist_ok=True)
    report_path = Path("results") / "dpo_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  💾 Saved → {report_path}")
    return report


# ─── STANDALONE RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    metrics_path = Path("results") / "three_stage_metrics.json"
    if not metrics_path.exists():
        print("Error: results/three_stage_metrics.json not found.")
        print("Run train_sft_dpo.py first.")
        exit(1)

    with open(metrics_path) as f:
        data = json.load(f)

    print("Loading outputs from three_stage_metrics.json...")
    sft_per = data["sft_model"]["per_test"]
    dpo_per = data["dpo_model"]["per_test"]

    sft_outputs = [t["output_preview"] for t in sft_per]
    dpo_outputs = [t["output_preview"] for t in dpo_per]

    generate_dpo_report(sft_outputs, dpo_outputs)
    print("\nDone.")
