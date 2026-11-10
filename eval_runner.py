import os
import json
import asyncio
import time
import yaml
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from groq import AsyncGroq
from dotenv import load_dotenv
from db import save_eval_run, get_previous_run, get_recent_runs
from diff_engine import compare_runs , detect_drift
from html_reporter import generate_html_report
from slack_alert import send_slack_alert
import  sys

load_dotenv()

# --- 1. ENHANCED SCHEMAS ---

class TestResult(BaseModel):
    test_id: str
    email_text: str
    expected_category: str
    expected_summary: str
    predicted_category: Optional[str] = None
    predicted_summary: Optional[str] = None
    
    # Multi-dimensional metrics
    category_match: bool = False
    summary_score: int = Field(default=0, description="LLM-as-judge score from 1 to 5")
    judge_reasoning: Optional[str] = None
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    error: Optional[str] = None

class EvalRunReport(BaseModel):
    prompt_version: str
    dataset_version: str
    timestamp: float
    total_tests: int
    passed_categories: int
    average_summary_score: float
    average_latency_seconds: float
    total_token_usage: int
    results: List[TestResult]


# --- 2. LLM-AS-A-JUDGE FOR SUMMARY RELEVANCE ---

JUDGE_PROMPT_TEMPLATE = """
You are an objective AI evaluation judge. Your task is to evaluate the quality and relevance of a generated summary compared to the expected summary and the original customer support email.

Customer Email:
---
{email_text}
---

Expected Summary (Ground Truth):
{expected_summary}

Generated Summary to Evaluate:
{predicted_summary}

Rate the generated summary on a scale of 1 to 5:
1 = Completely irrelevant, hallucinatory, or wrong.
2 = Poor relevance, misses the core issue.
3 = Acceptable, captures some essence but misses key details.
4 = Good, accurate, and concise summary of the issue.
5 = Excellent, perfectly captures the customer's core request with high accuracy.

Respond ONLY with a valid JSON object containing the keys "score" (an integer from 1 to 5) and "reasoning" (a short explanation).
"""

async def evaluate_summary_with_judge(
    client: AsyncGroq, 
    email_text: str, 
    expected_summary: str, 
    predicted_summary: Optional[str]
) -> Dict[str, Any]:
    """Uses an LLM judge to grade summary quality from 1 to 5."""
    if not predicted_summary:
        return {"score": 1, "reasoning": "No summary was generated due to an error."}
        
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        email_text=email_text,
        expected_summary=expected_summary,
        predicted_summary=predicted_summary
    )
    
    try:
        response = await client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "You are a strict, objective evaluation judge that outputs JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "score": int(result.get("score", 1)),
            "reasoning": result.get("reasoning", "No reasoning provided.")
        }
    except Exception as e:
        return {"score": 1, "reasoning": f"Judge evaluation failed: {str(e)}"}


# --- 3. ASYNC EVALUATION RUNNER WITH METRICS ---

async def run_evaluation(prompt_version: str = "v1.yaml", dataset_filename: str = "v1.0.0.json") -> EvalRunReport:
    # Load Prompt Config
    prompt_path = os.path.join("prompts", prompt_version)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_config = yaml.safe_load(f)

    # Load Dataset
    dataset_path = os.path.join("datasets", dataset_filename)
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    client = AsyncGroq()
    semaphore = asyncio.Semaphore(5)
    
    print(f"Starting multi-dimensional eval run for Prompt [{prompt_version}]...")
    print(f"Total test cases: {len(dataset['test_cases'])}\n")

    async def evaluate_single(tc: Dict[str, Any]) -> TestResult:
        async with semaphore:
            formatted_prompt = prompt_config["user_prompt_template"].format(email_text=tc["email_text"])
            
            start_time = time.time()
            pred_cat, pred_sum = None, None
            p_tokens, c_tokens, t_tokens = 0, 0, 0
            error_msg = None
            
            # 1. Run Classification
            try:
                response = await client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": prompt_config["system_prompt"]},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0
                )
                latency = time.time() - start_time
                
                # Extract token usage metadata from Groq response
                if response.usage:
                    p_tokens = response.usage.prompt_tokens
                    c_tokens = response.usage.completion_tokens
                    t_tokens = response.usage.total_tokens
                
                parsed = json.loads(response.choices[0].message.content)
                pred_cat = parsed.get("category")
                pred_sum = parsed.get("summary")
                
            except Exception as e:
                latency = time.time() - start_time
                error_msg = str(e)

            # 2. Check Binary Category Match
            category_match = False
            if pred_cat and pred_cat.strip().lower() == tc["expected_category"].strip().lower():
                category_match = True

            # 3. Run LLM-as-a-Judge for Summary Relevance (1-5 scale)
            judge_res = await evaluate_summary_with_judge(
                client, 
                tc["email_text"], 
                tc["expected_summary"], 
                pred_sum
            )

            return TestResult(
                test_id=tc["id"],
                email_text=tc["email_text"],
                expected_category=tc["expected_category"],
                expected_summary=tc["expected_summary"],
                predicted_category=pred_cat,
                predicted_summary=pred_sum,
                category_match=category_match,
                summary_score=judge_res["score"],
                judge_reasoning=judge_res["reasoning"],
                latency_seconds=round(latency, 3),
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                error=error_msg
            )

    tasks = [evaluate_single(tc) for tc in dataset["test_cases"]]
    results: List[TestResult] = await asyncio.gather(*tasks)

    # Compute aggregate metrics
    total = len(results)
    passed_cats = sum(1 for r in results if r.category_match)
    avg_summary_score = round(sum(r.summary_score for r in results) / total, 2) if total > 0 else 0.0
    avg_latency = round(sum(r.latency_seconds for r in results) / total, 3) if total > 0 else 0.0
    total_tokens = sum(r.total_tokens for r in results)

    report = EvalRunReport(
        prompt_version=prompt_version,
        dataset_version=dataset["dataset_version"],
        timestamp=time.time(),
        total_tests=total,
        passed_categories=passed_cats,
        average_summary_score=avg_summary_score,
        average_latency_seconds=avg_latency,
        total_token_usage=total_tokens,
        results=results
    )

    return report


# --- 4. EXECUTION ENTRYPOINT ---

if __name__ == "__main__":
    # 1. Run the evaluation suite asynchronously
    report = asyncio.run(run_evaluation(prompt_version="v1.yaml", dataset_filename="v1.0.0.json"))
    report_dict = report.dict()
    
    # 2. Fetch baseline and rolling history (last 15 runs for drift analysis)
    prev_run = get_previous_run(report.prompt_version)
    recent_runs = get_recent_runs(report.prompt_version, limit=15)
    
    # 3. Save current run to SQLite
    run_id = save_eval_run(report_dict)
    print(f"\n💾 Saved evaluation report to SQLite with Run ID: #{run_id}")
    
    # 4. Perform Diff Analysis & Slow Drift Detection
    diff = compare_runs(
        current_report=report_dict, 
        previous_report=prev_run,
        warning_threshold=3.0,
        critical_threshold=8.0
    )
    
    drift_analysis = detect_drift(recent_runs, window_size=7, drift_drop_threshold=4.0)
    
    # 5. Generate HTML Regression Report
    html_file = generate_html_report(report_dict, diff, recent_runs, output_filename="eval_report.html")
    print(f"📊 HTML Regression Report generated successfully: {html_file}")
    
    # 6. Push Notification to Slack
    try:
        send_slack_alert(report_dict, diff, report_url=f"file:///{os.path.abspath(html_file)}")
    except Exception:
        pass

    # --- CI/CD GATE: BLOCK MERGE ON CRITICAL REGRESSION ---
    if diff.get("status") == "compared" and diff.get("severity") == "CRITICAL":
        print(f"\n❌ BUILD FAILED: Critical regression detected ({diff['pass_rate_delta_percent']}% drop). Blocking PR merge.")
        sys.exit(1)
    else:
        print(f"\n✅ BUILD PASSED: Prompt evaluation check successful.")
        sys.exit(0)
    # --- REPORT 1: MULTI-DIMENSIONAL SUMMARY ---
    print("=" * 60)
    print("📊 MULTI-DIMENSIONAL EVALUATION REPORT")
    print("=" * 60)
    print(f"Prompt Version:       {report.prompt_version}")
    print(f"Dataset Version:      {report.dataset_version}")
    print(f"Category Accuracy:    {report.passed_categories}/{report.total_tests} ({round((report.passed_categories/report.total_tests)*100, 1)}%)")
    print(f"Avg Summary Score:    {report.average_summary_score} / 5.0")
    print(f"Avg Latency:          {report.average_latency_seconds}s")
    print(f"Total Token Usage:    {report.total_token_usage} tokens")
    print("=" * 60)
    
    # --- REPORT 2: REGRESSION DIFF & DRIFT ANALYSIS ---
    print("📈 REGRESSION DETECTION & DRIFT ANALYSIS")
    print("=" * 60)
    if diff["status"] == "baseline":
        print(f"ℹ️ {diff['message']}")
    else:
        print(f"Severity Status:    [{diff['severity']}]")
        print(f"Policy Evaluation:  {diff['alert_message']}")
        print(f"Overall Pass Rate:  {diff['current_pass_rate']}% (Prev: {diff['previous_pass_rate']}%)")
        print(f"Pass Rate Delta:    {diff['pass_rate_delta_percent']:+.1f}%")
        
        # Print Drift Status
        print(f"\n🐢 Drift Status:      [{'DRIFT DETECTED' if drift_analysis['drift_detected'] else 'STABLE'}]")
        print(f"Drift Message:      {drift_analysis['message']}")
        
        print("\nPer-Category Accuracy Shifts:")
        for cat, data in diff["category_deltas"].items():
            print(f" - {cat.upper()}: {data['current_accuracy']}% (Delta: {data['delta_percent']:+.1f}%)")
            
        print(f"\n🚨 Regressions Detected: {diff['regressions_count']}")
        for reg in diff["regressions"]:
            print(f"   * [{reg['test_id']}] {reg['email_snippet']}")
            
        print(f"\n🎉 Improvements Detected: {diff['improvements_count']}")
        for imp in diff["improvements"]:
            print(f"   * [{imp['test_id']}] {imp['email_snippet']}")
    print("=" * 60)
    
    # --- REPORT 3: DETAILED TEST BREAKDOWN ---
    print("\n🔍 Detailed Test Breakdown:")
    for r in report.results:
        status = "✅ PASS" if r.category_match else "❌ FAIL"
        print(f"[{r.test_id}] {status} | Cat Match: {r.category_match} | Summary Score: {r.summary_score}/5 | Latency: {r.latency_seconds}s | Tokens: {r.total_tokens}")
        if not r.category_match or r.summary_score < 3:
            print(f"   -> Expected Cat: '{r.expected_category}' | Got: '{r.predicted_category}'")
            print(f"   -> Judge Note: {r.judge_reasoning}")