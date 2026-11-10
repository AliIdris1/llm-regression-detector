import sqlite3
import json
from typing import Optional, Dict, Any

DB_NAME = "eval_history.db"

def init_db():
    """Initializes the SQLite database table for evaluation runs."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            prompt_version TEXT,
            dataset_version TEXT,
            pass_rate REAL,
            report_json TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_eval_run(report_dict: Dict[str, Any]) -> int:
    """Saves an evaluation report to SQLite and returns the run ID."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    total = report_dict["total_tests"]
    passed = report_dict["passed_categories"]
    pass_rate = (passed / total) if total > 0 else 0.0
    
    cursor.execute("""
        INSERT INTO eval_runs (timestamp, prompt_version, dataset_version, pass_rate, report_json)
        VALUES (?, ?, ?, ?, ?)
    """, (
        report_dict["timestamp"],
        report_dict["prompt_version"],
        report_dict["dataset_version"],
        pass_rate,
        json.dumps(report_dict)
    ))
    
    run_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return run_id

def get_previous_run(prompt_version: str) -> Optional[Dict[str, Any]]:
    """Retrieves the most recent previous evaluation run for a given prompt version."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # FIXED: Removed OFFSET 1 so it grabs the immediate last run
    cursor.execute("""
        SELECT report_json FROM eval_runs 
        WHERE prompt_version = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    """, (prompt_version,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return json.loads(row[0])
    return None

def get_recent_runs(prompt_version: str, limit: int = 10) -> list:
    """Retrieves the last N evaluation runs for trend analysis."""
    init_db()
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, pass_rate, report_json FROM eval_runs 
        WHERE prompt_version = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (prompt_version, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    runs = []
    for row in rows:
        runs.append({
            "timestamp": row[0],
            "pass_rate": row[1],
            "report": json.loads(row[2])
        })
    # Return in chronological order for charting
    return list(reversed(runs))