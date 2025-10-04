"""
Batch processing utilities for multiple DCRM CSV files.
"""

import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from datetime import datetime

from dcrm_ai_diagnostics.src.models.infer import predict_with_anomaly_from_df
from dcrm_ai_diagnostics.src.utils.database import log_analysis
from dcrm_ai_diagnostics.src.utils.report import generate_pdf_report


def process_batch_csvs(csv_files: List[bytes], file_names: List[str]) -> Dict[str, Any]:
    """
    Process multiple CSV files in batch.
    
    Args:
        csv_files: List of CSV file contents as bytes
        file_names: List of corresponding file names
    
    Returns:
        Dictionary with batch processing results
    """
    results = {
        "total_files": len(csv_files),
        "processed_files": 0,
        "failed_files": 0,
        "predictions": {},
        "summary": {},
        "errors": []
    }
    
    prediction_counts = {"Healthy": 0, "Worn Arcing Contact": 0, "Misaligned Mechanism": 0}
    health_scores = []
    anomaly_count = 0
    
    for i, (csv_content, file_name) in enumerate(zip(csv_files, file_names)):
        try:
            # Read CSV
            df = pd.read_csv(pd.io.common.BytesIO(csv_content))
            
            # Process with inference
            result = predict_with_anomaly_from_df(df)
            
            # Update counts
            label_map = {0: "Healthy", 1: "Worn Arcing Contact", 2: "Misaligned Mechanism"}
            prediction_label = label_map.get(result["prediction"], str(result["prediction"]))
            prediction_counts[prediction_label] += 1
            
            if result.get("health_score"):
                health_scores.append(result["health_score"])
            
            if result.get("anomaly"):
                anomaly_count += 1
            
            # Store individual result
            results["predictions"][file_name] = {
                "prediction": prediction_label,
                "health_score": result.get("health_score", 0),
                "anomaly": result.get("anomaly", False),
                "arcing_contact_health": result.get("component_insights", {}).get("arcing_contact_health", "Unknown"),
                "main_contact_health": result.get("component_insights", {}).get("main_contact_health", "Unknown"),
                "mechanism_health": result.get("component_insights", {}).get("mechanism_health", "Unknown"),
            }
            
            # Log to database
            try:
                log_analysis(
                    file_name=file_name,
                    file_source="batch_upload",
                    prediction=result["prediction"],
                    prediction_label=prediction_label,
                    anomaly=result.get("anomaly"),
                    anomaly_score=result.get("anomaly_score"),
                    health_score=result.get("health_score", 0),
                    component_insights=result.get("component_insights", {}),
                    top_features=result.get("top_features"),
                    maintenance_recommendations=result.get("maintenance_recommendations", [])
                )
            except Exception as e:
                results["errors"].append(f"Database logging failed for {file_name}: {str(e)}")
            
            results["processed_files"] += 1
            
        except Exception as e:
            results["failed_files"] += 1
            results["errors"].append(f"Processing failed for {file_name}: {str(e)}")
    
    # Generate summary
    results["summary"] = {
        "prediction_counts": prediction_counts,
        "average_health_score": sum(health_scores) / len(health_scores) if health_scores else 0,
        "anomaly_percentage": (anomaly_count / results["processed_files"]) * 100 if results["processed_files"] > 0 else 0,
        "processing_timestamp": datetime.now().isoformat()
    }
    
    return results


def generate_batch_report(results: Dict[str, Any], output_path: Path) -> Path:
    """
    Generate a comprehensive batch analysis report.
    
    Args:
        results: Batch processing results
        output_path: Path to save the report
    
    Returns:
        Path to the generated report
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create detailed report content
    title = f"Batch DCRM Analysis Report - {results['summary']['processing_timestamp'][:10]}"
    
    metadata = {
        "Generated At": results["summary"]["processing_timestamp"],
        "Total Files": results["total_files"],
        "Processed Files": results["processed_files"],
        "Failed Files": results["failed_files"],
        "Success Rate": f"{(results['processed_files'] / results['total_files']) * 100:.1f}%" if results["total_files"] > 0 else "0%"
    }
    
    summary = {
        "Prediction Distribution": f"Healthy: {results['summary']['prediction_counts']['Healthy']}, "
                                 f"Worn Arcing: {results['summary']['prediction_counts']['Worn Arcing Contact']}, "
                                 f"Misaligned: {results['summary']['prediction_counts']['Misaligned Mechanism']}",
        "Average Health Score": f"{results['summary']['average_health_score']:.1f}/100",
        "Anomaly Rate": f"{results['summary']['anomaly_percentage']:.1f}%",
        "Critical Issues": sum(1 for pred in results["predictions"].values() 
                              if pred["health_score"] < 40)
    }
    
    # Generate PDF report
    generated = generate_pdf_report(output_path, title, metadata, summary)
    
    return generated


def process_zip_file(zip_content: bytes) -> Dict[str, Any]:
    """
    Process a ZIP file containing multiple CSV files.
    
    Args:
        zip_content: ZIP file content as bytes
    
    Returns:
        Batch processing results
    """
    csv_files = []
    file_names = []
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract ZIP file
        zip_path = Path(temp_dir) / "batch_files.zip"
        with open(zip_path, "wb") as f:
            f.write(zip_content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.filelist:
                if file_info.filename.endswith('.csv'):
                    # Read CSV content
                    csv_content = zip_ref.read(file_info.filename)
                    csv_files.append(csv_content)
                    file_names.append(Path(file_info.filename).name)
    
    if not csv_files:
        return {
            "error": "No CSV files found in ZIP archive",
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0
        }
    
    return process_batch_csvs(csv_files, file_names)
