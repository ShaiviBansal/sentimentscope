import streamlit as st
import requests
import json

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="SentimentScope", page_icon="📈", layout="wide")

st.title("📈 SentimentScope")
st.markdown("Financial sentiment analysis — baseline vs fine-tuned DistilBERT with PEFT/LoRA")

# --- Prediction Section ---
st.header("Live Prediction")
text_input = st.text_input("Enter a financial headline:", placeholder="Apple stock surges after strong earnings...")

if st.button("Analyze") and text_input:
    with st.spinner("Analyzing..."):
        res = requests.post(f"{API_URL}/predict", json={"text": text_input})
        if res.status_code == 200:
            data = res.json()
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Baseline Model")
                b = data["baseline"]
                color = "🟢" if b["label"] == "Bullish" else "🔴" if b["label"] == "Bearish" else "🟡"
                st.metric("Prediction", f"{color} {b['label']}")
                st.metric("Confidence", f"{b['confidence']*100:.1f}%")
                st.bar_chart(b["probabilities"])
            with col2:
                st.subheader("Fine-Tuned Model (PEFT/LoRA)")
                f = data["finetuned"]
                color = "🟢" if f["label"] == "Bullish" else "🔴" if f["label"] == "Bearish" else "🟡"
                st.metric("Prediction", f"{color} {f['label']}")
                st.metric("Confidence", f"{f['confidence']*100:.1f}%")
                st.bar_chart(f["probabilities"])

# --- A/B Test Section ---
st.header("A/B Test Results")
st.markdown("Statistical comparison of baseline vs fine-tuned model on a sample batch.")

if st.button("Run A/B Test"):
    sample_texts = [
        "Company reports record profits beating analyst expectations",
        "Stock crashes after disappointing quarterly results",
        "Federal Reserve raises interest rates amid inflation concerns",
        "Tech giant announces massive layoffs amid restructuring",
        "Startup raises Series B funding to expand operations",
        "Market volatility increases as recession fears grow",
        "Earnings beat expectations sending shares higher",
        "CEO resigns amid accounting scandal investigation",
        "New product launch drives strong consumer demand",
        "Supply chain disruptions impact quarterly revenue"
    ]
    sample_labels = [1, 0, 2, 0, 1, 0, 1, 0, 1, 0]

    with st.spinner("Running A/B test..."):
        res = requests.post(f"{API_URL}/ab_test", json={
            "texts": sample_texts,
            "labels": sample_labels
        })
        if res.status_code == 200:
            d = res.json()
            col1, col2, col3 = st.columns(3)
            col1.metric("Baseline Accuracy", f"{d['baseline_accuracy']*100:.1f}%")
            col2.metric("Fine-tuned Accuracy", f"{d['finetuned_accuracy']*100:.1f}%")
            col3.metric("Improvement", f"+{d['improvement']*100:.1f}%")
            st.metric("p-value", f"{d['p_value']:.4f}")
            if d["significant"]:
                st.success("✅ Result is statistically significant (p < 0.05) — the improvement is real, not noise.")
            else:
                st.warning("⚠️ Result is not statistically significant yet — need more samples.")

# --- Drift Monitoring Section ---
st.header("Drift Monitor")
st.markdown("Tracks prediction distribution shift via Population Stability Index (PSI).")

if st.button("Check Drift"):
    res = requests.get(f"{API_URL}/drift")
    if res.status_code == 200:
        d = res.json()
        if "message" in d:
            st.info(d["message"])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Recent Distribution")
                st.bar_chart(d["recent_distribution"])
            with col2:
                st.subheader("Reference Distribution")
                st.bar_chart(d["reference_distribution"])
            psi = d["psi"]
            st.metric("PSI Score", f"{psi:.4f}")
            if d["drift_detected"]:
                st.error("🚨 Drift detected (PSI > 0.2) — model may need retraining.")
            else:
                st.success("✅ No significant drift detected.")