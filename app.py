import streamlit as st
import pandas as pd
import sqlite3
import os
import src.ml_models as ml
import src.dl_model as dl
import src.chatbot as cb

st.set_page_config(page_title="CustIntel Dashboard", page_icon="🤖", layout="wide")

st.sidebar.title("CustIntel Navigation")
page = st.sidebar.radio("Go to", ["Dashboard Home", "LTV & Churn Prediction", "Next Category Prediction", "GenAI Chatbot & Feedbacks"])

@st.cache_resource
def get_db_connection():
    return sqlite3.connect("data/ecommerce.db", check_same_thread=False)

def get_summary_metrics():
    conn = get_db_connection()
    stats = pd.read_sql_query("SELECT COUNT(*) as customers FROM Dim_Customers", conn)
    try:
        orders = pd.read_sql_query("SELECT COUNT(*) as orders FROM Fact_Order_Items", conn)
        revenue = pd.read_sql_query("SELECT SUM(price + freight_value) as revenue FROM Fact_Order_Items", conn)
        return stats['customers'][0], orders['orders'][0], revenue['revenue'][0]
    except:
        return stats['customers'][0], 0, 0.0

if page == "Dashboard Home":
    st.title("CustIntel - AI-Powered Customer Intelligence 🚀")
    st.write("Welcome to the ultimate E-Commerce AI Command Center.")
    
    col1, col2, col3 = st.columns(3)
    try:
        c, o, r = get_summary_metrics()
        col1.metric("Total Customers", f"{c:,}")
        col2.metric("Total Order Items", f"{o:,}")
        col3.metric("Total Revenue", f"${r:,.2f}")
    except Exception as e:
        st.warning(f"Database not initialized yet. Please run data ingestion. Error: {e}")

elif page == "LTV & Churn Prediction":
    st.title("Predict Customer Lifetime Value & Churn Risk")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Profile")
        freq = st.number_input("Purchase Frequency (Total Orders)", min_value=1, value=5)
        recency = st.number_input("Recency (Days since last purchase)", min_value=0, value=30)
        age = st.number_input("Customer Age (Days since first purchase)", min_value=0, value=365)
        spend = st.number_input("Total Historical Spend ($)", min_value=0.0, value=500.0)
        
    with col2:
        st.subheader("AI Prediction")
        if st.button("Predict Metrics"):
            try:
                churn_pred, churn_prob = ml.predict_churn(spend, freq, age)
                ltv_pred = ml.predict_ltv(freq, recency, age)
                
                st.metric("Predicted LTV", f"${ltv_pred:,.2f}")
                
                churn_color = "red" if churn_pred == 1 else "green"
                churn_text = "High Risk" if churn_pred == 1 else "Low Risk / Retained"
                st.markdown(f"**Churn Risk Level:** <span style='color:{churn_color}'>{churn_text}</span>", unsafe_allow_html=True)
                st.progress(float(churn_prob), text=f"Probability of Churning: {churn_prob*100:.1f}%")
                
            except Exception as e:
                st.error(f"Models are not trained yet! Error: {e}")

elif page == "Next Category Prediction":
    st.title("PyTorch Deep Learning: Next Purchase Intent")
    st.write("Enter the customer's last clicked or purchased product categories to predict their next move.")
    
    st.info("💡 Hint: Enter 1 to 5 categories separated by commas. (e.g. 'sports_leisure, bed_bath_table, health_beauty')")
    user_input = st.text_input("Customer Browsing History:", "sports_leisure, furniture_decor")
    
    if st.button("Predict Next Category"):
        cats = [c.strip() for c in user_input.split(',')]
        try:
            results = dl.predict_next_category(cats)
            st.write("### AI Propensity Engine Top 3 Recommendations")
            for i, res in enumerate(results):
                st.metric(f"Rank {i+1}: {res['category']}", f"{res['probability']*100:.1f}% confidence")
        except Exception as e:
            st.error(f"PyTorch Model not ready! Error: {e}")
            
elif page == "GenAI Chatbot & Feedbacks":
    st.title("Voice of Customer & Conversational BI")
    
    api_key_status = cb.init_genai()
    if not api_key_status:
        st.error("🔑 GEMINI_API_KEY is not set. Please set the environment variable and restart Streamlit to use GenAI features.")
        st.stop()
        
    tab1, tab2 = st.tabs(["💬 Analytics Chatbot", "📝 Product Summarizer"])
    
    with tab1:
        st.subheader("Chat with your database")
        st.write("Ask natural language questions like: 'How many customers are from sao paulo?' or 'What are the top 5 product categories?'")
        query = st.text_input("Ask a question:")
        if st.button("Ask AI"):
            with st.spinner("AI is thinking..."):
                summary, df = cb.sql_chatbot(query)
                st.write(f"**AI:** {summary}")
                if df is not None and not df.empty:
                    st.dataframe(df.head(10))
                    
    with tab2:
        st.subheader("Feedback Sentiment Summarizer")
        st.write("Summarizes reviews into actionable insights.")
        limit = st.slider("Number of random reviews to sample", 10, 100, 30)
        if st.button("Generate Summary"):
            with st.spinner("Reading reviews & synthesizing..."):
                summary = cb.summarize_reviews(limit=limit)
                st.write(summary)
