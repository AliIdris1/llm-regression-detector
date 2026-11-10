import os
import requests
from typing import Dict, Any

def send_slack_alert(
    current_report: Dict[str, Any], 
    diff_result: Dict[str, Any], 
    report_url: str = "http://localhost:8000/eval_report.html"
) -> bool:
    """
    Sends a structured, rich-block Slack alert via an incoming webhook URL.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        print("ℹ️ SLACK_WEBHOOK_URL not found in environment variables. Skipping Slack notification.")
        return False

    prompt_version = current_report.get("prompt_version", "v1.yaml")
    severity = diff_result.get("severity", "NORMAL")
    
    # Handle initial baseline run case
    if diff_result.get("status") == "baseline":
        text_summary = f"✨ Baseline evaluation saved for prompt `{prompt_version}`."
        color_emoji = "🔵"
    else:
        curr_rate = diff_result.get("current_pass_rate", 0.0)
        prev_rate = diff_result.get("previous_pass_rate", 0.0)
        delta = diff_result.get("pass_rate_delta_percent", 0.0)
        reg_count = diff_result.get("regressions_count", 0)
        
        if severity == "CRITICAL":
            color_emoji = "🚨"
        elif severity == "WARNING":
            color_emoji = "⚠️"
        else:
            color_emoji = "✅"
            
        text_summary = f"{color_emoji} *Prompt Evaluation Alert [{severity}]* — `{prompt_version}`\n" \
                       f"• *Pass Rate:* `{curr_rate}%` (Prev: `{prev_rate}%` | Delta: `{delta:+.1f}%`)\n" \
                       f"• *Regressions Detected:* `{reg_count}`"

    # Build Slack Block Kit Payload
    payload = {
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{color_emoji} AI Model Regression Report ({prompt_version})",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": text_summary
                }
            },
            {
                "type": "divider"
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Dataset Version:*\n{current_report.get('dataset_version', 'v1.0.0.json')}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Avg Summary Score:*\n{current_report.get('average_summary_score', 0)} / 5.0"
                    }
                ]
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "📊 View Full HTML Diff Report",
                            "emoji": True
                        },
                        "url": report_url,
                        "style": "primary"
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            print("🚀 Slack alert sent successfully!")
            return True
        else:
            print(f"❌ Failed to send Slack alert. Status: {response.status_code}, Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to Slack webhook: {str(e)}")
        return False