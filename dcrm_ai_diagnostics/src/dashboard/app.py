import io
from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# Ensure project root is on sys.path so 'dcrm_ai_diagnostics' package resolves
THIS_FILE = Path(__file__).resolve()
# app.py is .../SIH25189/dcrm_ai_diagnostics/src/dashboard/app.py
# We need the repo root (SIH25189) on sys.path → parents[4]
REPO_ROOT = THIS_FILE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dcrm_ai_diagnostics.src.models.infer import predict_from_df, ensure_smoothed, predict_with_anomaly_from_df, load_iforest
from dcrm_ai_diagnostics.src.utils.report import generate_pdf_report
from dcrm_ai_diagnostics.src.utils.database import get_analysis_history, get_health_trends
from dcrm_ai_diagnostics.src.utils.batch_processor import process_zip_file, generate_batch_report
from datetime import datetime
from io import BytesIO


st.set_page_config(page_title="DCRM Diagnostics", layout="wide")
st.title("DCRM Diagnostics Dashboard")

st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose page", ["Analysis", "Batch Processing", "History", "Health Trends"])

# Initialize variables
limit = 50
days = 30
uploaded = None
use_sample = False
sample_path = Path("dcrm_ai_diagnostics/data/processed/sample_dcrm.csv")

if page == "Analysis":
    st.sidebar.header("Upload DCRM CSV")
    uploaded = st.sidebar.file_uploader("Choose CSV", type=["csv"]) 
    use_sample = st.sidebar.button("Load sample")
elif page == "History":
    st.sidebar.header("Analysis History")
    limit = st.sidebar.slider("Number of records", 10, 100, 50)
elif page == "Batch Processing":
    st.sidebar.header("Batch Processing")
    st.sidebar.info("Upload a ZIP file containing multiple CSV files for batch analysis.")
elif page == "Health Trends":
    st.sidebar.header("Health Trends")
    days = st.sidebar.slider("Days to look back", 7, 90, 30)

if page == "Analysis":
    df = None
    if uploaded is not None:
        df = pd.read_csv(uploaded)
    elif use_sample and sample_path.exists():
        df = pd.read_csv(sample_path)

    if df is not None:
        df = ensure_smoothed(df)
        st.subheader("Waveform")
        fig = go.Figure()
        if "time" in df.columns and "resistance" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["resistance"], name="resistance", line=dict(color="#888")))
        if "time" in df.columns and "resistance_smooth" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["resistance_smooth"], name="smoothed", line=dict(color="#d62728")))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Prediction & Anomaly")
        try:
            result = predict_with_anomaly_from_df(df)
            label_map = {0: "Healthy", 1: "Worn Arcing Contact", 2: "Misaligned Mechanism"}
            st.write("Prediction:", label_map.get(result["prediction"], str(result["prediction"])))
            if result["probabilities"] is not None:
                st.write("Class probabilities:", result["probabilities"][0])
            if result["anomaly"] is not None:
                st.write("Anomaly:", "Yes" if result["anomaly"] else "No")
                if result["anomaly_score"] is not None:
                    st.write("Anomaly score (higher=more normal):", result["anomaly_score"])

            # Health Score
            if result.get("health_score") is not None:
                health_score = result["health_score"]
                st.subheader(f"Overall Health Score: {health_score}/100")
                if health_score >= 80:
                    st.success("✅ Excellent condition")
                elif health_score >= 60:
                    st.warning("⚠️ Good condition with minor issues")
                elif health_score >= 40:
                    st.error("❌ Fair condition - maintenance required")
                else:
                    st.error("🚨 Poor condition - immediate attention needed")

            # Component Analysis
            if result.get("component_insights"):
                st.subheader("Component Analysis")
                insights = result["component_insights"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    status = insights.get("arcing_contact_health", "Unknown")
                    if status == "Good":
                        st.success(f"Arcing Contact: {status}")
                    elif status == "Degraded":
                        st.warning(f"Arcing Contact: {status}")
                    else:
                        st.error(f"Arcing Contact: {status}")
                
                with col2:
                    status = insights.get("main_contact_health", "Unknown")
                    if status == "Good":
                        st.success(f"Main Contact: {status}")
                    elif status == "Degraded":
                        st.warning(f"Main Contact: {status}")
                    else:
                        st.error(f"Main Contact: {status}")
                
                with col3:
                    status = insights.get("mechanism_health", "Unknown")
                    if status == "Good":
                        st.success(f"Mechanism: {status}")
                    elif status == "Degraded":
                        st.warning(f"Mechanism: {status}")
                    else:
                        st.error(f"Mechanism: {status}")

            # RUL Estimation
            if result.get("rul_estimation"):
                st.subheader("Remaining Useful Life (RUL) Estimation")
                rul = result["rul_estimation"]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("RUL (Days)", rul["estimated_rul_days"])
                with col2:
                    st.metric("RUL (Weeks)", rul["estimated_rul_weeks"])
                with col3:
                    st.metric("RUL (Months)", rul["estimated_rul_months"])
                with col4:
                    urgency = rul["urgency_level"]
                    if urgency == "CRITICAL":
                        st.error(f"Urgency: {urgency}")
                    elif urgency == "HIGH":
                        st.warning(f"Urgency: {urgency}")
                    elif urgency == "MEDIUM":
                        st.info(f"Urgency: {urgency}")
                    else:
                        st.success(f"Urgency: {urgency}")
                
                st.write(f"**Next Inspection Date:** {rul['next_inspection_date']}")
                st.write(f"**Confidence:** {rul['confidence_percentage']:.1f}%")
                
                # Risk factors
                if rul.get("risk_factors"):
                    st.write("**Risk Factors:**")
                    for risk in rul["risk_factors"]:
                        st.write(f"- {risk}")

            # Maintenance Recommendations
            if result.get("maintenance_recommendations"):
                st.subheader("Maintenance Recommendations")
                for i, rec in enumerate(result["maintenance_recommendations"], 1):
                    if "URGENT" in rec or "CRITICAL" in rec:
                        st.error(f"{i}. {rec}")
                    elif "ANOMALY" in rec:
                        st.warning(f"{i}. {rec}")
                    else:
                        st.info(f"{i}. {rec}")
                
                # RUL-specific recommendations
                if result.get("rul_estimation") and result["rul_estimation"].get("maintenance_recommendations"):
                    st.write("**RUL-based Recommendations:**")
                    for i, rec in enumerate(result["rul_estimation"]["maintenance_recommendations"], 1):
                        if "URGENT" in rec or "CRITICAL" in rec:
                            st.error(f"{i}. {rec}")
                        elif "Priority" in rec:
                            st.warning(f"{i}. {rec}")
                        else:
                            st.info(f"{i}. {rec}")

            # SHAP top features
            if result.get("top_features"):
                st.subheader("Top contributing features")
                for item in result["top_features"]:
                    st.write(f"{item['name']}: {item['shap_value']:+.4f}")

            # PDF export section
            st.subheader("Export Report")
            title = "DCRM Diagnostic Report"
            metadata = {
                "Generated At": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Source": "Uploaded" if uploaded is not None else ("Sample" if use_sample else "Unknown"),
            }
            summary = {
                "Prediction": label_map.get(result["prediction"], str(result["prediction"])),
                "Anomaly": "Yes" if (result.get("anomaly") is True) else ("No" if result.get("anomaly") is False else "N/A"),
                "Anomaly Score": None if result.get("anomaly_score") is None else round(float(result["anomaly_score"]), 5),
            }

            # Render current chart to PNG (simple export using Plotly)
            img_bytes = None
            try:
                img_bytes = fig.to_image(format="png")  # requires kaleido (Plotly)
            except Exception:
                img_bytes = None

            out_path = Path("dcrm_ai_diagnostics/reports/diagnostic_report.pdf")
            
            # Generate PDF and show download button
            try:
                generated = generate_pdf_report(out_path, title, metadata, summary, image_bytes=img_bytes)
                with open(generated, "rb") as f:
                    st.download_button(
                        label="Download PDF Report",
                        data=f.read(),
                        file_name="diagnostic_report.pdf",
                        mime="application/pdf",
                    )
            except Exception as e:
                st.error(f"PDF generation failed: {str(e)}")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload a CSV with columns like 'time' and 'resistance', or click Load sample.")

elif page == "Batch Processing":
    st.subheader("Batch DCRM Analysis")
    
    uploaded_zip = st.file_uploader("Upload ZIP file containing CSV files", type=["zip"])
    
    if uploaded_zip is not None:
        try:
            # Process ZIP file
            zip_content = uploaded_zip.read()
            results = process_zip_file(zip_content)
            
            if "error" in results:
                st.error(results["error"])
            else:
                # Display summary
                st.subheader("Batch Processing Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Files", results["total_files"])
                with col2:
                    st.metric("Processed", results["processed_files"])
                with col3:
                    st.metric("Failed", results["failed_files"])
                with col4:
                    success_rate = (results["processed_files"] / results["total_files"]) * 100 if results["total_files"] > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Prediction distribution
                if results["summary"]:
                    st.subheader("Prediction Distribution")
                    pred_counts = results["summary"]["prediction_counts"]
                    fig = px.pie(values=list(pred_counts.values()), names=list(pred_counts.keys()), 
                               title="Prediction Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Health score statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Health Score", f"{results['summary']['average_health_score']:.1f}/100")
                    with col2:
                        st.metric("Anomaly Rate", f"{results['summary']['anomaly_percentage']:.1f}%")
                    with col3:
                        critical_count = sum(1 for pred in results["predictions"].values() if pred["health_score"] < 40)
                        st.metric("Critical Issues", critical_count)
                
                # Individual file results
                if results["predictions"]:
                    st.subheader("Individual File Results")
                    df_results = pd.DataFrame.from_dict(results["predictions"], orient='index')
                    df_results.index.name = 'File Name'
                    st.dataframe(df_results, use_container_width=True)
                
                # Errors
                if results["errors"]:
                    st.subheader("Processing Errors")
                    for error in results["errors"]:
                        st.error(error)
                
                # Generate batch report
                if st.button("Generate Batch Report"):
                    report_path = Path("dcrm_ai_diagnostics/reports/batch_report.pdf")
                    try:
                        generated_path = generate_batch_report(results, report_path)
                        with open(generated_path, "rb") as f:
                            st.download_button(
                                label="Download Batch Report",
                                data=f.read(),
                                file_name="batch_analysis_report.pdf",
                                mime="application/pdf",
                            )
                        st.success("Batch report generated successfully!")
                    except Exception as e:
                        st.error(f"Report generation failed: {str(e)}")
        
        except Exception as e:
            st.error(f"Batch processing failed: {str(e)}")
    else:
        st.info("Upload a ZIP file containing multiple CSV files for batch analysis.")

elif page == "History":
    st.subheader("Analysis History")
    
    try:
        history = get_analysis_history(limit)
        
        if history:
            # Create a DataFrame for display
            df_history = pd.DataFrame(history)
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyses", len(df_history))
            with col2:
                healthy_count = len(df_history[df_history['prediction_label'] == 'Healthy'])
                st.metric("Healthy", healthy_count)
            with col3:
                worn_count = len(df_history[df_history['prediction_label'] == 'Worn Arcing Contact'])
                st.metric("Worn Arcing Contact", worn_count)
            with col4:
                misaligned_count = len(df_history[df_history['prediction_label'] == 'Misaligned Mechanism'])
                st.metric("Misaligned Mechanism", misaligned_count)
            
            # Display recent analyses
            st.subheader("Recent Analyses")
            for record in history[:10]:  # Show last 10
                with st.expander(f"Analysis {record['id']} - {record['timestamp']} - {record['prediction_label']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**File:** {record['file_name']}")
                        st.write(f"**Health Score:** {record['health_score']}/100")
                        st.write(f"**Anomaly:** {'Yes' if record['anomaly'] else 'No'}")
                    with col2:
                        st.write(f"**Arcing Contact:** {record['arcing_contact_health']}")
                        st.write(f"**Main Contact:** {record['main_contact_health']}")
                        st.write(f"**Mechanism:** {record['mechanism_health']}")
                    
                    if record.get('maintenance_recommendations'):
                        st.write("**Recommendations:**")
                        for rec in record['maintenance_recommendations'][:3]:  # Show first 3
                            st.write(f"- {rec}")
        else:
            st.info("No analysis history found. Run some analyses first.")
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")

elif page == "Health Trends":
    st.subheader("Health Score Trends")
    
    try:
        trends = get_health_trends(days)
        
        if trends:
            # Create DataFrame for plotting
            df_trends = pd.DataFrame(trends)
            df_trends['timestamp'] = pd.to_datetime(df_trends['timestamp'])
            
            # Plot health score over time
            fig = px.line(df_trends, x='timestamp', y='health_score', 
                         title=f'Health Score Trend (Last {days} Days)',
                         labels={'health_score': 'Health Score', 'timestamp': 'Date'})
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show recent health scores
            st.subheader("Recent Health Scores")
            recent_df = df_trends.tail(10)[['timestamp', 'health_score', 'prediction_label', 'file_name']]
            st.dataframe(recent_df, use_container_width=True)
            
            # Health score statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Health Score", f"{df_trends['health_score'].mean():.1f}")
            with col2:
                st.metric("Min Health Score", f"{df_trends['health_score'].min():.1f}")
            with col3:
                st.metric("Max Health Score", f"{df_trends['health_score'].max():.1f}")
        else:
            st.info("No health trend data found. Run some analyses first.")
    except Exception as e:
        st.error(f"Error loading trends: {str(e)}")