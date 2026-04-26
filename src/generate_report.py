import nbformat as nbf
import os

def generate_notebook():
    nb = nbf.v4.new_notebook()

    text_intro = """# CustIntel Project Report: Advanced EDA, ML & Deep Learning
This notebook contains Exploratory Data Analysis (EDA) visualizations, summary of the Classical ML models (Churn/LTV), OLS diagnostics, and PyTorch Deep Learning validation curves plotted directly from the processed dataset (`data/ecommerce.db`).
"""
    code_setup = """import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
%matplotlib inline

sns.set_theme(style="whitegrid")

conn = sqlite3.connect('../data/ecommerce.db')
df_customers = pd.read_sql_query("SELECT * FROM Dim_Customers", conn)
df_fact = pd.read_sql_query("SELECT * FROM Fact_Order_Items WHERE order_purchase_timestamp IS NOT NULL", conn)
df_products = pd.read_sql_query("SELECT * FROM Dim_Products", conn)
conn.close()

df_agg = df_fact.groupby('customer_id').agg(
    total_spend=('price', 'sum'),
    frequency=('order_id', 'nunique')
).reset_index()"""

    text_eda = "## 1. Exploratory Data Analysis (EDA)\n### 1.1 Box Plots for Outlier Detection (Whale Spenders)"
    code_eda1 = """plt.figure(figsize=(10, 6))
sns.boxplot(x=df_agg['total_spend'])
plt.title('Box Plot of Total Customer Spend (Outlier Detection)')
plt.xlabel('Total Spend ($)')
plt.show()"""

    text_eda2 = "### 1.2 Category Sales Distribution (Stacked Bar Plot)"
    code_eda2 = """# Stacked Bar Plot: Sales by Customer State for Top 5 Categories
top_5_cats = df_products['product_category_name'].value_counts().head(5).index
merged_df = df_fact.merge(df_products, on='product_id').merge(df_customers, on='customer_id')
stacked_data = merged_df[merged_df['product_category_name'].isin(top_5_cats)]
pivot_df = stacked_data.groupby(['customer_state', 'product_category_name']).size().unstack().fillna(0)
pivot_df = pivot_df.loc[pivot_df.sum(axis=1).sort_values(ascending=False).head(10).index]

fig, ax = plt.subplots(figsize=(12, 7))
pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
plt.title('Stacked Bar Plot: Sales of Top 5 Categories by State')
plt.ylabel('Order Count')
plt.xlabel('Customer State')
plt.tight_layout()
plt.show()"""

    text_eda3 = "### 1.3 Scatter Plot: Frequency vs Total Spend"
    code_eda3 = """plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_agg, x='frequency', y='total_spend', alpha=0.5)
plt.title('Scatter Plot: Purchase Frequency vs. Total Spend')
plt.xlabel('Number of Orders (Frequency)')
plt.ylabel('Total Spend ($)')
plt.show()"""

    text_ml = "## 2. Classical Machine Learning Metrics & Diagnostics\n### 2.1 OLS Diagnostics for LTV Regressor"
    code_ml = """try:
    with open('../models/ols_summary.txt', 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("Run `python src/ml_models.py` first to generate OLS diagnostics.")"""

    text_dl = "## 3. Deep Learning (PyTorch) Validation\n### 3.1 Loss Curve for Next-Category Predictor"
    code_dl = """try:
    with open('../models/dl_history.json', 'r') as f:
        history = json.load(f)
        
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Validation Loss')
    plt.title('PyTorch MLP Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()
except FileNotFoundError:
    print("Run `python src/dl_model.py` first to generate DL history.")"""

    nb['cells'] = [
        nbf.v4.new_markdown_cell(text_intro),
        nbf.v4.new_code_cell(code_setup),
        nbf.v4.new_markdown_cell(text_eda),
        nbf.v4.new_code_cell(code_eda1),
        nbf.v4.new_markdown_cell(text_eda2),
        nbf.v4.new_code_cell(code_eda2),
        nbf.v4.new_markdown_cell(text_eda3),
        nbf.v4.new_code_cell(code_eda3),
        nbf.v4.new_markdown_cell(text_ml),
        nbf.v4.new_code_cell(code_ml),
        nbf.v4.new_markdown_cell(text_dl),
        nbf.v4.new_code_cell(code_dl)
    ]

    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/Project_Report.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Project_Report.ipynb created successfully in notebooks/ directory.")

if __name__ == "__main__":
    generate_notebook()
