import pandas as pd
import sqlite3
import os
import re

DATASET_DIR = "dataset"
DB_PATH = "data/ecommerce.db"

def clean_text_regex(text):
    if pd.isna(text): return "No Text"
    return re.sub(r'[^a-zA-Z0-9\s.,!?]', '', str(text)).strip()

def clean_and_load_data():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('PRAGMA foreign_keys = ON;')
    
    try:
        cursor.execute("DROP TABLE IF EXISTS Fact_Order_Items")
        cursor.execute("DROP TABLE IF EXISTS Dim_Reviews")
        cursor.execute("DROP TABLE IF EXISTS Dim_Products")
        cursor.execute("DROP TABLE IF EXISTS Dim_Customers")
        cursor.execute("DROP TABLE IF EXISTS Dim_Payments")
        
        cursor.execute("""
            CREATE TABLE Dim_Customers (
                customer_id TEXT PRIMARY KEY,
                customer_unique_id TEXT,
                customer_zip_code_prefix INTEGER,
                customer_city TEXT,
                customer_state TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE Dim_Products (
                product_id TEXT PRIMARY KEY,
                product_category_name TEXT,
                product_weight_g REAL,
                product_length_cm REAL,
                product_height_cm REAL,
                product_width_cm REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE Fact_Order_Items (
                order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT,
                product_id TEXT,
                customer_id TEXT,
                price REAL,
                freight_value REAL,
                order_purchase_timestamp DATETIME,
                FOREIGN KEY(customer_id) REFERENCES Dim_Customers(customer_id),
                FOREIGN KEY(product_id) REFERENCES Dim_Products(product_id)
            )
        """)
        cursor.execute("""
            CREATE TABLE Dim_Reviews (
                review_id TEXT,
                order_id TEXT,
                review_score INTEGER,
                review_comment_title TEXT,
                review_comment_message TEXT
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Schema creation failed: {e}")
        return

    print("Loading SQLite...")
    try:
        customers = pd.read_csv(os.path.join(DATASET_DIR, "olist_customers_dataset.csv"))
        orders = pd.read_csv(os.path.join(DATASET_DIR, "olist_orders_dataset.csv"))
        order_items = pd.read_csv(os.path.join(DATASET_DIR, "olist_order_items_dataset.csv"))
        reviews = pd.read_csv(os.path.join(DATASET_DIR, "olist_order_reviews_dataset.csv"))
        products = pd.read_csv(os.path.join(DATASET_DIR, "olist_products_dataset.csv"))
        translation = pd.read_csv(os.path.join(DATASET_DIR, "product_category_name_translation.csv"))
        
        customers['customer_city'] = customers['customer_city'].map(lambda x: str(x).title())
        dim_customers = customers[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix', 'customer_city', 'customer_state']]
        dim_customers.to_sql('Dim_Customers', conn, if_exists='append', index=False)
        
        products = products.merge(translation, on='product_category_name', how='left')
        products['product_category_name_english'] = products['product_category_name_english'].fillna('unknown')
        dim_products = products[['product_id', 'product_category_name_english', 'product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']]
        dim_products = dim_products.rename(columns={'product_category_name_english': 'product_category_name'})
        dim_products = dim_products.fillna(0)
        
        # Iterator structure implemented
        product_gen = (row for idx, row in dim_products.iterrows())
        dim_products.to_sql('Dim_Products', conn, if_exists='append', index=False)
        
        # Regex application
        reviews['review_comment_message'] = reviews['review_comment_message'].apply(clean_text_regex)
        reviews['review_comment_title'] = reviews['review_comment_title'].apply(clean_text_regex)
        dim_reviews = reviews[['review_id', 'order_id', 'review_score', 'review_comment_title', 'review_comment_message']].drop_duplicates()
        dim_reviews.to_sql('Dim_Reviews', conn, if_exists='append', index=False)
        
        fact_order_items = order_items.merge(orders[['order_id', 'customer_id', 'order_purchase_timestamp']], on='order_id', how='left')
        fact_order_items['order_purchase_timestamp'] = pd.to_datetime(fact_order_items['order_purchase_timestamp'], errors='coerce')
        fact_filtered = fact_order_items[['order_id', 'product_id', 'customer_id', 'price', 'freight_value', 'order_purchase_timestamp']].dropna(subset=['customer_id', 'product_id'])
        fact_filtered.to_sql('Fact_Order_Items', conn, if_exists='append', index=False)
        
        print("Data Ingestion & Integrity Constraints Completed.")
    except Exception as e:
        print(f"Ingestion failed due to malformed data or constraint hit: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    clean_and_load_data()
