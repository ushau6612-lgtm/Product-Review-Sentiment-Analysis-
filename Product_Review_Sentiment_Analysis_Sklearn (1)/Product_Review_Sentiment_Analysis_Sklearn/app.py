import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# Robust Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "sentiment_model.pkl").exists() and (BASE_DIR / "Product_Review_Sentiment_Analysis_Sklearn" / "sentiment_model.pkl").exists():
    BASE_DIR = BASE_DIR / "Product_Review_Sentiment_Analysis_Sklearn"

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-positive {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .badge-neutral {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.3);
    }
    .badge-negative {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    .sentiment-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💬 Product Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify customer feedback into <b>Positive</b>, <b>Neutral</b>, or <b>Negative</b> sentiments using a trained <b>TF-IDF + LogisticRegression</b> NLP pipeline.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "sentiment_model.pkl"
csv_path = BASE_DIR / "data" / "product_reviews.csv"
chart_path = BASE_DIR / "sentiment_distribution.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Single & Batch Sentiment Analyzer", "📈 Sentiment Distribution Plot", "📋 Reviews Dataset"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Customer Review Input")
        
        sample_options = {
            "Custom text...": "",
            "🌟 'This product is excellent and works perfectly!'": "This product is excellent and works perfectly, exceeding all my expectations!",
            "⭐ 'Good value for money and easy to use'": "Good value for money and very easy to use every day.",
            "⚖️ 'It is okay, nothing special'": "It is okay, nothing special. Works fine but quality could be improved.",
            "⚠️ 'Not worth the price, very disappointing'": "Not worth the price, very disappointing. Stopped working after 2 days."
        }
        
        selected_sample = st.selectbox("⚡ Load Pre-written Customer Review", list(sample_options.keys()))
        default_text = sample_options[selected_sample] if selected_sample != "Custom text..." else "This product is fantastic! High quality and super fast shipping."
        
        with st.form("sentiment_form"):
            review_text = st.text_area("Customer Review Content", value=default_text, height=130, placeholder="Paste customer review here...")
            submit_btn = st.form_submit_button("🚀 Analyze Sentiment", use_container_width=True)
            
    with col_result:
        st.subheader("Sentiment Analysis Verdict")
        if submit_btn and review_text.strip():
            prediction = model.predict([review_text])[0]
            prob_arr = model.predict_proba([review_text])[0]
            classes = list(model.classes_)
            
            # Map probabilities to classes
            prob_map = {cls: prob_arr[i] for i, cls in enumerate(classes)}
            confidence = prob_map[prediction]
            
            if prediction == "positive":
                st.markdown(f"""
                <div class="badge-positive">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Detected Sentiment</div>
                    <div class="sentiment-title">🌟 POSITIVE SENTIMENT</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == "negative":
                st.markdown(f"""
                <div class="badge-negative">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Detected Sentiment</div>
                    <div class="sentiment-title">⚠️ NEGATIVE SENTIMENT</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-neutral">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Detected Sentiment</div>
                    <div class="sentiment-title">⚖️ NEUTRAL SENTIMENT</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Confidence: {confidence:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📊 Class Probability Breakdown")
            c_pos = prob_map.get("positive", 0.0)
            c_neu = prob_map.get("neutral", 0.0)
            c_neg = prob_map.get("negative", 0.0)
            
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.metric("Positive", f"{c_pos:.1%}")
            col_p2.metric("Neutral", f"{c_neu:.1%}")
            col_p3.metric("Negative", f"{c_neg:.1%}")
            
            st.progress(float(c_pos))
            st.caption("Green bar indicates Positive likelihood score")
        elif submit_btn:
            st.warning("Please type or paste a review text first.")
        else:
            st.info("👈 Enter a review or select a preset and click **'Analyze Sentiment'**.")
            
    st.write("---")
    st.subheader("⚡ Bulk Review Batch Testing")
    with st.expander("Evaluate multiple reviews simultaneously"):
        bulk_input = st.text_area(
            "Enter multiple reviews (one per line):",
            "This product is outstanding and exceeded expectations!\nWorst customer experience ever, completely broken on arrival.\nIt's fine, does what it says on the box.\nAffordable price and fast delivery.",
            height=120
        )
        if st.button("Analyze Batch Reviews"):
            lines = [l.strip() for l in bulk_input.split("\n") if l.strip()]
            if lines:
                preds = model.predict(lines)
                probs = model.predict_proba(lines).max(axis=1)
                batch_df = pd.DataFrame({
                    "Review": lines,
                    "Sentiment": [p.upper() for p in preds],
                    "Confidence": [f"{pr:.1%}" for pr in probs]
                })
                st.dataframe(batch_df, use_container_width=True)

with tab2:
    st.subheader("Sentiment Distribution in Training Set")
    if chart_path.exists():
        st.image(str(chart_path), caption="Training Reviews Sentiment Breakdown", use_container_width=True)
    else:
        st.info("Distribution plot not found. Run 'python train_model.py' to generate.")

with tab3:
    st.subheader("Customer Reviews Corpus (product_reviews.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", f"{len(df):,}")
        col2.metric("Positive Reviews", f"{(df['sentiment'] == 'positive').sum():,}")
        col3.metric("Negative Reviews", f"{(df['sentiment'] == 'negative').sum():,}")
        
        filter_sentiment = st.multiselect("Filter by Sentiment", options=list(df['sentiment'].unique()), default=list(df['sentiment'].unique()))
        filtered_df = df[df['sentiment'].isin(filter_sentiment)]
        st.dataframe(filtered_df, use_container_width=True)
    else:
        st.warning("Dataset file not found.")
