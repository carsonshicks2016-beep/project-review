"""Step 3.10 — Evaluation Script.

Runs the eval_set.json against the configured DEEP_READ_PROVIDER (Gemini)
to measure accuracy, calibration, and quote faithfulness.
"""

import json
import time
from pathlib import Path
from bellwether import pipeline, config
from bellwether.logging_setup import get_logger

log = get_logger("bellwether.eval")

def run_eval():
    eval_path = Path("data/eval_set.json")
    if not eval_path.exists():
        log.error("Eval set not found at %s", eval_path)
        return
        
    with open(eval_path) as f:
        dataset = json.load(f)
        
    provider = config.require("DEEP_READ_PROVIDER")
    log.info("Starting evaluation using provider: %s", provider)
    
    results = []
    correct_type = 0
    correct_direction = 0
    hallucinations = 0
    total_material = 0
    
    for i, item in enumerate(dataset):
        log.info("Evaluating item %d/%d: %s", i+1, len(dataset), item["id"])
        
        # We run the deep read directly for the eval
        t0 = time.time()
        signal = pipeline.run_deep_read(item["text"])
        duration = time.time() - t0
        
        expected = item["expected"]
        expected_material = expected["is_material"]
        
        predicted_material = signal is not None
        
        result_entry = {
            "id": item["id"],
            "expected_material": expected_material,
            "predicted_material": predicted_material,
            "duration": round(duration, 2)
        }
        
        if predicted_material:
            # Check accuracy
            type_match = signal.catalyst_type == expected.get("catalyst_type")
            dir_match = signal.direction == expected.get("direction")
            
            if type_match: correct_type += 1
            if dir_match: correct_direction += 1
            
            # Check hallucination: is the quote actually in the text?
            # We strip whitespace to be slightly forgiving of newline differences, 
            # but it should be an exact substring match.
            clean_quote = " ".join(signal.evidence_quote.split())
            clean_text = " ".join(item["text"].split())
            
            is_hallucination = clean_quote not in clean_text
            if is_hallucination:
                hallucinations += 1
                
            total_material += 1
            
            result_entry.update({
                "predicted_type": signal.catalyst_type,
                "predicted_direction": signal.direction,
                "type_match": type_match,
                "dir_match": dir_match,
                "hallucinated": is_hallucination,
                "quote": signal.evidence_quote
            })
            
        results.append(result_entry)
        
        # Respect Gemini free tier limits (~15 RPM)
        log.info("Sleeping for 4 seconds to respect rate limits...")
        time.sleep(4)
        
    # Generate Markdown Report
    report = [
        "# Evaluation Results",
        f"\n**Provider:** `{provider}`",
        f"**Total Samples:** {len(dataset)}",
        "\n## Summary Metrics",
    ]
    
    if total_material > 0:
        report.extend([
            f"- **Type Accuracy:** {correct_type}/{total_material} ({(correct_type/total_material)*100:.1f}%)",
            f"- **Direction Accuracy:** {correct_direction}/{total_material} ({(correct_direction/total_material)*100:.1f}%)",
            f"- **Quote Hallucinations:** {hallucinations}/{total_material} ({(hallucinations/total_material)*100:.1f}%)",
        ])
    
    report.append("\n## Detailed Results\n")
    report.append("| ID | Exp Material | Pred Material | Type Match | Dir Match | Hallucinated | Latency |")
    report.append("|---|---|---|---|---|---|---|")
    
    for r in results:
        t_match = "✅" if r.get("type_match") else ("❌" if r.get("predicted_material") else "-")
        d_match = "✅" if r.get("dir_match") else ("❌" if r.get("predicted_material") else "-")
        halluc = "⚠️ YES" if r.get("hallucinated") else ("✅ NO" if r.get("predicted_material") else "-")
        
        report.append(
            f"| {r['id']} | {r['expected_material']} | {r['predicted_material']} | "
            f"{t_match} | {d_match} | {halluc} | {r['duration']}s |"
        )
        
    report_path = Path("eval_results.md")
    report_path.write_text("\n".join(report))
    log.info("Eval complete. Report written to %s", report_path.absolute())

if __name__ == "__main__":
    run_eval()
