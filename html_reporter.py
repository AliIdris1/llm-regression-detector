import os
import datetime
from typing import Dict, Any, List

def generate_html_report(
    current_report: Dict[str, Any], 
    diff_result: Dict[str, Any], 
    recent_runs: List[Dict[str, Any]], 
    output_filename: str = "eval_report.html"
) -> str:
    """
    Generates a professional, self-contained HTML regression report complete with metadata,
    scorecard, side-by-side regression diffs, and a historical trend chart.
    """
    # Format timestamp
    ts = current_report.get("timestamp", datetime.datetime.now().timestamp())
    formatted_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    
    prompt_version = current_report.get("prompt_version", "v1.yaml")
    dataset_version = current_report.get("dataset_version", "v1.0.0.json")
    
    total = current_report.get("total_tests", 0)
    passed = current_report.get("passed_categories", 0)
    pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0
    
    # Diff metrics
    prev_pass_rate = diff_result.get("previous_pass_rate", pass_rate)
    pass_delta = diff_result.get("pass_rate_delta_percent", 0.0)
    severity = diff_result.get("severity", "NORMAL")
    alert_msg = diff_result.get("alert_message", "Stable")
    regressions = diff_result.get("regressions", [])
    
    # Prepare chart data
    chart_labels = []
    chart_data = []
    for run in recent_runs:
        t_str = datetime.datetime.fromtimestamp(run["timestamp"]).strftime('%m-%d %H:%M')
        chart_labels.append(t_str)
        chart_data.append(round(run["pass_rate"] * 100, 1))

    # HTML Template
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eval Regression Report - {prompt_version}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-950 text-slate-100 font-sans antialiased min-h-screen p-6">
    <div class="max-w-6xl mx-auto space-y-6">
        
        <!-- Header & Metadata -->
        <header class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
            <div>
                <div class="flex items-center gap-3">
                    <span class="px-3 py-1 bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 rounded-full text-xs font-semibold uppercase tracking-wider">AI Regression Report</span>
                    <span class="px-3 py-1 bg-slate-800 text-slate-300 rounded-full text-xs font-medium">Severity: <strong class="{'text-red-400' if severity in ['WARNING', 'CRITICAL'] else 'text-emerald-400'}">{severity}</strong></span>
                </div>
                <h1 class="text-2xl font-bold mt-2 tracking-tight text-white">Prompt Evaluation & Diff Analysis</h1>
                <p class="text-sm text-slate-400 mt-1">{alert_msg}</p>
            </div>
            <div class="bg-slate-950/60 border border-slate-800/80 rounded-lg p-4 text-xs space-y-1.5 w-full md:w-auto">
                <div class="flex justify-between gap-6"><span class="text-slate-400">Prompt Version:</span> <span class="font-mono text-indigo-300">{prompt_version}</span></div>
                <div class="flex justify-between gap-6"><span class="text-slate-400">Dataset Version:</span> <span class="font-mono text-slate-200">{dataset_version}</span></div>
                <div class="flex justify-between gap-6"><span class="text-slate-400">Model:</span> <span class="font-mono text-slate-200">openai/gpt-oss-20b</span></div>
                <div class="flex justify-between gap-6"><span class="text-slate-400">Timestamp:</span> <span class="font-mono text-slate-400">{formatted_date}</span></div>
            </div>
        </header>

        <!-- Scorecard Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Pass Rate</p>
                <div class="flex items-baseline gap-2 mt-2">
                    <span class="text-3xl font-extrabold text-white">{pass_rate}%</span>
                    <span class="text-xs font-semibold {'text-emerald-400' if pass_delta >= 0 else 'text-red-400'}">{'+' if pass_delta >= 0 else ''}{pass_delta}%</span>
                </div>
                <p class="text-xs text-slate-500 mt-1">Previous: {prev_pass_rate}%</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Summary Quality (Judge)</p>
                <div class="flex items-baseline gap-2 mt-2">
                    <span class="text-3xl font-extrabold text-white">{current_report.get("average_summary_score", 0)}</span>
                    <span class="text-xs text-slate-400">/ 5.0</span>
                </div>
                <p class="text-xs text-slate-500 mt-1">LLM-as-a-judge score</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Avg Latency</p>
                <div class="flex items-baseline gap-2 mt-2">
                    <span class="text-3xl font-extrabold text-white">{current_report.get("average_latency_seconds", 0)}s</span>
                </div>
                <p class="text-xs text-slate-500 mt-1">Per request average</p>
            </div>
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 shadow-lg">
                <p class="text-xs font-medium text-slate-400 uppercase tracking-wider">Total Token Usage</p>
                <div class="flex items-baseline gap-2 mt-2">
                    <span class="text-3xl font-extrabold text-white">{format(current_report.get("total_token_usage", 0), ",d")}</span>
                </div>
                <p class="text-xs text-slate-500 mt-1">Prompt + Completion</p>
            </div>
        </div>

        <!-- Trend Chart Section -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <h2 class="text-lg font-semibold text-white mb-4">Historical Pass Rate Trend ({prompt_version})</h2>
            <div class="h-64 w-full">
                <canvas id="trendChart"></canvas>
            </div>
        </div>

        <!-- Regressions Side-by-Side Table -->
        <div class="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-xl">
            <div class="flex justify-between items-center mb-4">
                <h2 class="text-lg font-semibold text-white">Regressed Cases ({len(regressions)})</h2>
                <span class="text-xs text-slate-400">Cases that flipped from Pass to Fail</span>
            </div>
    """

    if not regressions:
        html_content += """
            <div class="p-8 text-center bg-slate-950/40 rounded-lg border border-slate-800/60 text-slate-400">
                🎉 No regressions detected in this run! Your prompt improvements are holding steady.
            </div>
        """
    else:
        html_content += """
            <div class="overflow-x-auto">
                <table class="w-full text-left border-collapse">
                    <thead>
                        <tr class="border-b border-slate-800 text-xs text-slate-400 uppercase tracking-wider">
                            <th class="py-3 px-4">Test ID</th>
                            <th class="py-3 px-4">Email Snippet</th>
                            <th class="py-3 px-4">Previous Result</th>
                            <th class="py-3 px-4">Current Result</th>
                            <th class="py-3 px-4">Reason / Note</th>
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-800 text-sm">
        """
        for reg in regressions:
            html_content += f"""
                        <tr class="hover:bg-slate-800/40 transition">
                            <td class="py-4 px-4 font-mono text-indigo-400 font-semibold">{reg.get('test_id')}</td>
                            <td class="py-4 px-4 text-slate-300 max-w-xs truncate">{reg.get('email_snippet')}</td>
                            <td class="py-4 px-4"><span class="px-2.5 py-1 bg-emerald-500/10 text-emerald-400 rounded text-xs font-medium">Passed</span></td>
                            <td class="py-4 px-4"><span class="px-2.5 py-1 bg-red-500/10 text-red-400 rounded text-xs font-medium">Failed</span></td>
                            <td class="py-4 px-4 text-slate-400 text-xs">{reg.get('reason')}</td>
                        </tr>
            """
        html_content += """
                    </tbody>
                </table>
            </div>
        """

    html_content += f"""
        </div>
    </div>

    <!-- Chart.js Script Setup -->
    <script>
        const ctx = document.getElementById('trendChart').getContext('2d');
        const trendChart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {chart_labels},
                datasets: [{{
                    label: 'Pass Rate (%)',
                    data: {chart_data},
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3,
                    pointBackgroundColor: '#6366f1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{
                        min: 0,
                        max: 100,
                        grid: {{ color: 'rgba(255, 255, 255, 0.05)' }},
                        ticks: {{ color: '#94a3b8' }}
                    }},
                    x: {{
                        grid: {{ display: false }},
                        ticks: {{ color: '#94a3b8' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Save to disk
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return output_filename