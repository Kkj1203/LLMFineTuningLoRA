"""
evaluate.py — Before/After Fine-Tuning Evaluation

Measures JSON extraction quality:
  - Exact match: is the output valid parseable JSON?
  - Field accuracy: what fraction of expected fields are present and correct?
  - Format compliance: does the output contain ONLY JSON (no extra text)?

Run after training to compare base model vs fine-tuned model.
Results saved to results/metrics.json
"""

import json
import re
from pathlib import Path

# ─── TEST CASES (held-out, not in training data) ───────────────────────────────
TEST_CASES = [
    {
        "input": "Customer: Alex Turner. Order #55821. 2x noise-cancelling headphones at $249 each. Shipping to 88 Pine Road, Seattle. Express delivery, 2 days.",
        "expected": {
            "customer_name": "Alex Turner",
            "order_number": "55821",
            "item": "noise-cancelling headphones",
            "quantity": 2,
            "unit_price": 249,
            "shipping_address": "88 Pine Road, Seattle",
            "delivery_days": 2
        }
    },
    {
        "input": "Event: Annual Tech Summit. Date: September 20-22. Venue: Moscone Center, San Francisco. Expected attendance: 5,000. Registration deadline: August 31. Fee: $299.",
        "expected": {
            "event_name": "Annual Tech Summit",
            "start_date": "September 20",
            "end_date": "September 22",
            "venue": "Moscone Center",
            "city": "San Francisco",
            "expected_attendance": 5000,
            "registration_deadline": "August 31",
            "fee": 299
        }
    },
    {
        "input": "Software package: numpy version 1.26.4. Language: Python. License: BSD. Downloads last month: 18M. Dependencies: none. Supports Python 3.9-3.12.",
        "expected": {
            "package_name": "numpy",
            "version": "1.26.4",
            "language": "Python",
            "license": "BSD",
            "monthly_downloads": 18000000,
            "python_versions": ["3.9", "3.10", "3.11", "3.12"]
        }
    },
    {
        "input": "Hotel booking: Marriott Downtown. Room: Deluxe King. Check-in: Oct 5. Check-out: Oct 8. Rate: $189/night. Total: $567. Guest: Nina Patel. Confirmation: HTL-29841.",
        "expected": {
            "hotel": "Marriott Downtown",
            "room_type": "Deluxe King",
            "check_in": "October 5",
            "check_out": "October 8",
            "nightly_rate": 189,
            "total": 567,
            "guest_name": "Nina Patel",
            "confirmation_number": "HTL-29841"
        }
    },
    {
        "input": "Pull request #342 by dev-john. Repo: backend-api. Title: Add rate limiting middleware. Status: Open. Files changed: 4. Lines added: 87. Lines removed: 12. Reviewers: alice, bob.",
        "expected": {
            "pr_number": 342,
            "author": "dev-john",
            "repository": "backend-api",
            "title": "Add rate limiting middleware",
            "status": "Open",
            "files_changed": 4,
            "lines_added": 87,
            "lines_removed": 12,
            "reviewers": ["alice", "bob"]
        }
    }
]

PROMPT_TEMPLATE = """<|user|>
Extract structured JSON from the following text. Output ONLY valid JSON, nothing else.

Text: {input}
<|end|>
<|assistant|>
"""

# ─── SCORING ──────────────────────────────────────────────────────────────────

def is_valid_json(text: str) -> bool:
    """Check if text is parseable JSON."""
    try:
        # Strip any markdown code blocks
        cleaned = re.sub(r'```json\s*|\s*```', '', text).strip()
        json.loads(cleaned)
        return True
    except Exception:
        return False

def parse_json_safe(text: str) -> dict:
    """Parse JSON from model output, handling common formatting issues."""
    try:
        cleaned = re.sub(r'```json\s*|\s*```', '', text).strip()
        # Find first { and last }
        start = cleaned.find('{')
        end   = cleaned.rfind('}')
        if start != -1 and end != -1:
            return json.loads(cleaned[start:end+1])
    except Exception:
        pass
    return {}

def field_accuracy(predicted: dict, expected: dict) -> float:
    """What fraction of expected fields are present with correct values."""
    if not expected:
        return 0.0
    correct = 0
    for key, exp_val in expected.items():
        if key in predicted:
            pred_val = predicted[key]
            # Flexible matching
            if isinstance(exp_val, (int, float)):
                try:
                    if abs(float(pred_val) - float(exp_val)) < 0.01:
                        correct += 1
                except Exception:
                    pass
            elif isinstance(exp_val, list):
                if isinstance(pred_val, list):
                    # Check overlap
                    exp_set  = set(str(v).lower() for v in exp_val)
                    pred_set = set(str(v).lower() for v in pred_val)
                    overlap  = len(exp_set & pred_set) / len(exp_set)
                    correct += overlap
            elif isinstance(exp_val, bool):
                if str(pred_val).lower() in [str(exp_val).lower(), "true" if exp_val else "false"]:
                    correct += 1
            else:
                if str(pred_val).lower().strip() == str(exp_val).lower().strip():
                    correct += 1
                elif str(exp_val).lower() in str(pred_val).lower():
                    correct += 0.5  # partial credit
    return round(correct / len(expected), 3)

def format_compliance(text: str) -> float:
    """Score 1.0 if output is ONLY JSON, 0.5 if JSON with extra text, 0.0 if no JSON."""
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        return 1.0
    elif '{' in stripped and '}' in stripped:
        return 0.5
    return 0.0

# ─── EVALUATION ───────────────────────────────────────────────────────────────

def evaluate_outputs(model_outputs: list[str], label: str = "model") -> dict:
    """
    Evaluate a list of model outputs against test cases.
    model_outputs: list of raw text outputs, one per test case.
    """
    assert len(model_outputs) == len(TEST_CASES), \
        f"Expected {len(TEST_CASES)} outputs, got {len(model_outputs)}"

    results = []
    for i, (output, test) in enumerate(zip(model_outputs, TEST_CASES)):
        parsed    = parse_json_safe(output)
        valid     = is_valid_json(output)
        accuracy  = field_accuracy(parsed, test["expected"])
        compliance= format_compliance(output)

        results.append({
            "test_id":          i + 1,
            "input_preview":    test["input"][:60] + "...",
            "valid_json":       valid,
            "field_accuracy":   accuracy,
            "format_compliance":compliance,
            "output_preview":   output[:100].strip(),
        })

        print(f"  [{i+1}] valid={valid} | accuracy={accuracy:.2f} | compliance={compliance:.1f}")

    avg_valid      = round(sum(r["valid_json"]       for r in results) / len(results), 3)
    avg_accuracy   = round(sum(r["field_accuracy"]   for r in results) / len(results), 3)
    avg_compliance = round(sum(r["format_compliance"]for r in results) / len(results), 3)
    overall        = round((avg_valid + avg_accuracy + avg_compliance) / 3, 3)

    summary = {
        "label":            label,
        "num_tests":        len(results),
        "valid_json_rate":  avg_valid,
        "field_accuracy":   avg_accuracy,
        "format_compliance":avg_compliance,
        "overall_score":    overall,
        "per_test":         results,
    }

    print(f"\n  {'─'*40}")
    print(f"  Valid JSON:    {avg_valid:.1%}")
    print(f"  Field Accuracy:{avg_accuracy:.1%}")
    print(f"  Format Score:  {avg_compliance:.1%}")
    print(f"  Overall:       {overall:.3f} / 1.000")

    return summary


def compare_and_save(before_outputs: list[str], after_outputs: list[str]):
    """Compare before and after fine-tuning, save to results/metrics.json"""

    print("\n" + "="*50)
    print("  BEFORE FINE-TUNING")
    print("="*50)
    before = evaluate_outputs(before_outputs, label="base_model")

    print("\n" + "="*50)
    print("  AFTER FINE-TUNING (LoRA)")
    print("="*50)
    after  = evaluate_outputs(after_outputs,  label="finetuned_model")

    improvement = {
        "valid_json_improvement":   round(after["valid_json_rate"]  - before["valid_json_rate"],  3),
        "field_accuracy_improvement": round(after["field_accuracy"] - before["field_accuracy"],   3),
        "format_improvement":       round(after["format_compliance"]- before["format_compliance"],3),
        "overall_improvement":      round(after["overall_score"]    - before["overall_score"],    3),
    }

    print("\n" + "="*50)
    print("  IMPROVEMENT SUMMARY")
    print("="*50)
    print(f"  Valid JSON:     {before['valid_json_rate']:.1%} → {after['valid_json_rate']:.1%}  (+{improvement['valid_json_improvement']:.1%})")
    print(f"  Field Accuracy: {before['field_accuracy']:.1%} → {after['field_accuracy']:.1%}  (+{improvement['field_accuracy_improvement']:.1%})")
    print(f"  Format:         {before['format_compliance']:.1%} → {after['format_compliance']:.1%}  (+{improvement['format_improvement']:.1%})")
    print(f"  Overall:        {before['overall_score']:.3f} → {after['overall_score']:.3f}  (+{improvement['overall_improvement']:.3f})")

    output = {
        "before": before,
        "after":  after,
        "improvement": improvement,
        "test_prompts": [PROMPT_TEMPLATE.format(input=t["input"]) for t in TEST_CASES],
    }

    Path("results").mkdir(exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n💾 Saved to results/metrics.json")
    print("   Commit this file to GitHub for your portfolio.")
    return output


# ─── STANDALONE RUN ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("This script is called from the Colab notebook.")
    print("It exports: TEST_CASES, PROMPT_TEMPLATE, compare_and_save()")
    print(f"Test cases loaded: {len(TEST_CASES)}")