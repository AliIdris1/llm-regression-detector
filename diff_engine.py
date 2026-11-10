from typing import Dict, Any, List, Optional


def detect_drift(
    recent_runs: List[Dict[str, Any]], 
    window_size: int = 7, 
    drift_drop_threshold: float = 4.0
) -> Dict[str, Any]:
    """
    Detects slow performance drift by comparing the rolling average of the last N runs
    against earlier historical performance.
    """
    if not recent_runs or len(recent_runs) < window_size:
        return {
            "drift_detected": False, 
            "message": f"Insufficient runs for drift detection (have {len(recent_runs)}, need at least {window_size})."
        }
    
    # Extract pass rates as percentages
    pass_rates = [r["pass_rate"] * 100 for r in recent_runs]
    
    # Current rolling window (last N runs)
    rolling_window = pass_rates[-window_size:]
    rolling_avg = sum(rolling_window) / len(rolling_window)
    
    # Earlier historical baseline (runs before the current rolling window)
    historical_window = pass_rates[:-window_size]
    if not historical_window:
        historical_avg = pass_rates[0]
    else:
        historical_avg = sum(historical_window) / len(historical_window)
        
    delta = rolling_avg - historical_avg
    
    drift_detected = False
    message = "No slow drift detected. Rolling performance is stable."
    
    if delta <= -drift_drop_threshold:
        drift_detected = True
        message = (
            f"🐢 SLOW DRIFT WARNING: 7-run moving average ({rolling_avg:.1f}%) "
            f"has drifted down by {abs(delta):.1f}% compared to historical baseline ({historical_avg:.1f}%)."
        )
        
    return {
        "drift_detected": drift_detected,
        "rolling_average": round(rolling_avg, 1),
        "historical_average": round(historical_avg, 1),
        "drift_delta": round(delta, 1),
        "message": message
    }

def compare_runs(
    current_report: Dict[str, Any], 
    previous_report: Optional[Dict[str, Any]],
    warning_threshold: float = 3.0,  # 3% drop triggers warning
    critical_threshold: float = 8.0  # 8% drop triggers critical alert
) -> Dict[str, Any]:
    """
    Compares the current evaluation report against the previous baseline report.
    Calculates overall deltas, category shifts, regressions, improvements, 
    and applies a statistical significance threshold policy (Signal vs Noise).
    """
    if not previous_report:
        return {"status": "baseline", "message": "No previous run found. Saved as baseline."}

    # 1. Overall Pass Rate Delta
    curr_total = current_report["total_tests"]
    curr_passed = current_report["passed_categories"]
    curr_pass_rate = (curr_passed / curr_total) if curr_total > 0 else 0.0

    prev_total = previous_report["total_tests"]
    prev_passed = previous_report["passed_categories"]
    prev_pass_rate = (prev_passed / prev_total) if prev_total > 0 else 0.0

    pass_rate_delta_percent = round((curr_pass_rate - prev_pass_rate) * 100, 2)

    # 2. Threshold Evaluation (Signal vs Noise)
    # Regressions are negative deltas. We measure the drop magnitude.
    drop_magnitude = abs(pass_rate_delta_percent) if pass_rate_delta_percent < 0 else 0.0
    
    severity = "NORMAL"
    alert_message = "Performance is stable or improved."

    if pass_rate_delta_percent < 0:
        if drop_magnitude >= critical_threshold:
            severity = "CRITICAL"
            alert_message = f"🚨 CRITICAL REGRESSION: Pass rate dropped by {drop_magnitude}%, exceeding the {critical_threshold}% critical threshold!"
        elif drop_magnitude >= warning_threshold:
            severity = "WARNING"
            alert_message = f"⚠️ WARNING: Pass rate dropped by {drop_magnitude}%, exceeding the {warning_threshold}% warning threshold."
        else:
            severity = "NOISE"
            alert_message = f"ℹ️ Minor fluctuation ({drop_magnitude}% drop) is below the {warning_threshold}% threshold (classified as noise)."

    # 3. Per-Category Accuracy Breakdown & Delta
    def get_category_stats(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        stats = {}
        for r in results:
            cat = r["expected_category"]
            if cat not in stats:
                stats[cat] = {"total": 0, "passed": 0}
            stats[cat]["total"] += 1
            if r["category_match"] and r["summary_score"] >= 3:
                stats[cat]["passed"] += 1
        return stats

    curr_cat_stats = get_category_stats(current_report["results"])
    prev_cat_stats = get_category_stats(previous_report["results"])

    category_deltas = {}
    all_categories = set(list(curr_cat_stats.keys()) + list(prev_cat_stats.keys()))
    
    for cat in all_categories:
        curr_s = curr_cat_stats.get(cat, {"total": 0, "passed": 0})
        prev_s = prev_cat_stats.get(cat, {"total": 0, "passed": 0})
        
        curr_acc = (curr_s["passed"] / curr_s["total"]) if curr_s["total"] > 0 else 0.0
        prev_acc = (prev_s["passed"] / prev_s["total"]) if prev_s["total"] > 0 else 0.0
        
        category_deltas[cat] = {
            "current_accuracy": round(curr_acc * 100, 1),
            "previous_accuracy": round(prev_acc * 100, 1),
            "delta_percent": round((curr_acc - prev_acc) * 100, 1)
        }

    # 4. Regressions and Improvements (Test Case Level Diff)
    prev_results_map = {r["test_id"]: r for r in previous_report["results"]}
    
    regressions = []
    improvements = []

    for curr_r in current_report["results"]:
        test_id = curr_r["test_id"]
        curr_success = curr_r["category_match"] and curr_r["summary_score"] >= 3
        
        if test_id in prev_results_map:
            prev_r = prev_results_map[test_id]
            prev_success = prev_r["category_match"] and prev_r["summary_score"] >= 3
            
            # Flipped from Pass -> Fail (Regression)
            if prev_success and not curr_success:
                regressions.append({
                    "test_id": test_id,
                    "email_snippet": curr_r["email_text"][:60] + "...",
                    "previous_category": prev_r.get("predicted_category"),
                    "current_category": curr_r.get("predicted_category"),
                    "reason": "Passed in previous run, but failed category or summary threshold in current run."
                })
            
            # Flipped from Fail -> Pass (Improvement)
            elif not prev_success and curr_success:
                improvements.append({
                    "test_id": test_id,
                    "email_snippet": curr_r["email_text"][:60] + "...",
                    "previous_category": prev_r.get("predicted_category"),
                    "current_category": curr_r.get("predicted_category"),
                    "reason": "Failed in previous run, but successfully passed in current run."
                })

    return {
        "status": "compared",
        "severity": severity,
        "alert_message": alert_message,
        "pass_rate_delta_percent": pass_rate_delta_percent,
        "current_pass_rate": round(curr_pass_rate * 100, 1),
        "previous_pass_rate": round(prev_pass_rate * 100, 1),
        "category_deltas": category_deltas,
        "regressions_count": len(regressions),
        "improvements_count": len(improvements),
        "regressions": regressions,
        "improvements": improvements
    }