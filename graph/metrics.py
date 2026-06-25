import json
import os

METRICS_FILE = "metrics.json"

def export_metrics(metrics: dict):
    """Exports performance metrics to metrics.json."""
    try:
        # Load existing metrics
        existing_metrics = []
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r") as f:
                try:
                    existing_metrics = json.load(f)
                except json.JSONDecodeError:
                    existing_metrics = []
        
        # Append new metrics
        existing_metrics.append(metrics)
        
        # Save
        with open(METRICS_FILE, "w") as f:
            json.dump(existing_metrics, f, indent=4)
    except Exception as e:
        print(f"Failed to export metrics: {e}")
