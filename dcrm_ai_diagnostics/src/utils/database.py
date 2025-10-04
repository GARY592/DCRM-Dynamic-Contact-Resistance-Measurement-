"""
SQLite database for logging DCRM analysis history.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


PROJECT_ROOT = Path("dcrm_ai_diagnostics")
DB_PATH = PROJECT_ROOT / "data" / "analysis_history.db"


def init_database():
    """Initialize the SQLite database with required tables."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create analysis history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            file_name TEXT,
            file_source TEXT,
            prediction INTEGER,
            prediction_label TEXT,
            anomaly BOOLEAN,
            anomaly_score REAL,
            health_score INTEGER,
            arcing_contact_health TEXT,
            main_contact_health TEXT,
            mechanism_health TEXT,
            top_features TEXT,
            component_risks TEXT,
            maintenance_recommendations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()


def log_analysis(
    file_name: str,
    file_source: str,
    prediction: int,
    prediction_label: str,
    anomaly: Optional[bool],
    anomaly_score: Optional[float],
    health_score: int,
    component_insights: Dict[str, Any],
    top_features: Optional[List[Dict[str, Any]]],
    maintenance_recommendations: List[str]
) -> int:
    """
    Log an analysis result to the database.
    
    Returns:
        Analysis ID
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Prepare data for insertion
    timestamp = datetime.now().isoformat()
    top_features_json = json.dumps(top_features) if top_features else None
    component_risks_json = json.dumps(component_insights.get("component_risks", []))
    maintenance_recs_json = json.dumps(maintenance_recommendations)
    
    cursor.execute("""
        INSERT INTO analysis_history (
            timestamp, file_name, file_source, prediction, prediction_label,
            anomaly, anomaly_score, health_score, arcing_contact_health,
            main_contact_health, mechanism_health, top_features,
            component_risks, maintenance_recommendations
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, file_name, file_source, prediction, prediction_label,
        anomaly, anomaly_score, health_score,
        component_insights.get("arcing_contact_health"),
        component_insights.get("main_contact_health"),
        component_insights.get("mechanism_health"),
        top_features_json, component_risks_json, maintenance_recs_json
    ))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return analysis_id


def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieve analysis history from the database.
    
    Args:
        limit: Maximum number of records to return
    
    Returns:
        List of analysis records
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM analysis_history 
        ORDER BY created_at DESC 
        LIMIT ?
    """, (limit,))
    
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    conn.close()
    
    # Convert rows to dictionaries
    history = []
    for row in rows:
        record = dict(zip(columns, row))
        # Parse JSON fields
        if record.get("top_features"):
            record["top_features"] = json.loads(record["top_features"])
        if record.get("component_risks"):
            record["component_risks"] = json.loads(record["component_risks"])
        if record.get("maintenance_recommendations"):
            record["maintenance_recommendations"] = json.loads(record["maintenance_recommendations"])
        history.append(record)
    
    return history


def get_analysis_by_id(analysis_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific analysis by ID.
    
    Args:
        analysis_id: ID of the analysis to retrieve
    
    Returns:
        Analysis record or None if not found
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM analysis_history WHERE id = ?
    """, (analysis_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    columns = [description[0] for description in cursor.description]
    record = dict(zip(columns, row))
    
    # Parse JSON fields
    if record.get("top_features"):
        record["top_features"] = json.loads(record["top_features"])
    if record.get("component_risks"):
        record["component_risks"] = json.loads(record["component_risks"])
    if record.get("maintenance_recommendations"):
        record["maintenance_recommendations"] = json.loads(record["maintenance_recommendations"])
    
    return record


def get_health_trends(days: int = 30) -> List[Dict[str, Any]]:
    """
    Get health score trends over time.
    
    Args:
        days: Number of days to look back
    
    Returns:
        List of health score records
    """
    init_database()
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT timestamp, health_score, prediction_label, file_name
        FROM analysis_history 
        WHERE created_at >= datetime('now', '-{} days')
        ORDER BY created_at ASC
    """.format(days))
    
    columns = [description[0] for description in cursor.description]
    rows = cursor.fetchall()
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]

