import os
import sqlite3
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_PATH = "data/ecommerce.db"

def init_genai():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set.")
        return False
    genai.configure(api_key=api_key)
    return True

def summarize_reviews(product_id=None, limit=50):
    if not init_genai():
        return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
        
    conn = sqlite3.connect(DB_PATH)
    if product_id:
        query = f"""
        SELECT r.review_score, r.review_comment_title, r.review_comment_message
        FROM Dim_Reviews r
        JOIN Fact_Order_Items f ON r.order_id = f.order_id
        WHERE f.product_id = '{product_id}' AND r.review_comment_message != 'No Review'
        LIMIT {limit}
        """
    else:
        query = f"""
        SELECT review_score, review_comment_title, review_comment_message
        FROM Dim_Reviews
        WHERE review_comment_message != 'No Review'
        ORDER BY RANDOM() LIMIT {limit}
        """
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        conn.close()
        return f"Database error: {e}"
        
    conn.close()
    
    if df.empty:
        return "No reviews found for this selection."
        
    reviews_text = "\n".join([f"Score: {row['review_score']}, Title: {row['review_comment_title']}, Comment: {row['review_comment_message']}" for _, row in df.iterrows()])
    
    prompt = f"""
    You are an AI assistant for the CustIntel E-commerce platform.
    Below are some customer reviews. 
    Analyze the sentiment, extract key themes, and summarize what customers are saying.
    
    Reviews:
    {reviews_text}
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error connecting to Gemini API: {e}"

def sql_chatbot(user_query):
    if not init_genai():
        return "Gemini API key not configured. Please set GEMINI_API_KEY environment variable.", None
        
    schema_info = """
    We have an SQLite database with the following schema:
    1. Dim_Customers: customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state
    2. Dim_Products: product_id, product_category_name, product_weight_g, product_length_cm, product_height_cm, product_width_cm
    3. Dim_Reviews: review_id, order_id, review_score, review_comment_title, review_comment_message, review_creation_date, review_answer_timestamp
    4. Fact_Order_Items: order_id, order_item_id, product_id, seller_id, shipping_limit_date, price, freight_value, customer_id, order_status, order_purchase_timestamp, order_approved_at, order_delivered_carrier_date, order_delivered_customer_date, order_estimated_delivery_date
    5. Dim_Payments: order_id, payment_sequential, payment_type, payment_installments, payment_value
    """
    
    prompt = f"""
    {schema_info}
    
    The user asks: "{user_query}"
    
    Output ONLY a valid SQL query to answer this question. Do not include markdown formatting like ```sql or explanations. Just the SQL query.
    Return ONLY a single SQL query.
    IMPORTANT RULES:
    1. For string comparisons in WHERE clauses (e.g., city names, status), ALWAYS make the comparison case-insensitive using LOWER(), for example: LOWER(customer_city) = LOWER('sao paulo').
    2. If the user asks a conversational question not related to the database schema, return "NOT_SQL".
    """
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        sql_query = response.text.strip()
        
        if sql_query == "NOT_SQL" or not sql_query.upper().startswith("SELECT"):
            # It's a general question
            chat_prompt = f"User asks: {user_query}. You are the CustIntel AI assistant. Answer politely."
            resp = model.generate_content(chat_prompt)
            return resp.text, None
            
        sql_query = sql_query.replace('```sql', '').replace('```', '').strip()
        
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(sql_query, conn)
        conn.close()
        
        res_summary_prompt = f"The user asked: '{user_query}'\nThe DB returned this data:\n{df.head(10).to_string()}\nSummarize the answer naturally without mentioning the SQL query."
        summary_resp = model.generate_content(res_summary_prompt)
        
        return summary_resp.text, df
    except Exception as e:
        return f"Error processing query: {e}", None

if __name__ == "__main__":
    pass
