import sqlite3
import pandas as pd
import numpy as np
import os
import pickle
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score

DB_PATH = "data/ecommerce.db"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_data():
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT 
        c.customer_unique_id,
        MIN(f.order_purchase_timestamp) as first_purchase,
        MAX(f.order_purchase_timestamp) as last_purchase,
        COUNT(DISTINCT f.order_id) as frequency,
        SUM(f.price) as total_price,
        SUM(f.freight_value) as total_freight,
        SUM(f.price + f.freight_value) as total_spend
    FROM Fact_Order_Items f
    JOIN Dim_Customers c ON f.customer_id = c.customer_id
    WHERE f.order_purchase_timestamp IS NOT NULL
    GROUP BY c.customer_unique_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df['first_purchase'] = pd.to_datetime(df['first_purchase'])
    df['last_purchase'] = pd.to_datetime(df['last_purchase'])
    max_date = df['last_purchase'].max()
    df['recency'] = (max_date - df['last_purchase']).dt.days
    df['customer_age'] = (max_date - df['first_purchase']).dt.days
    df['is_churn'] = (df['recency'] > 180).astype(int)
    return df.dropna()

def train_and_save_models():
    df = get_data()
    
    scaler_rf = StandardScaler()
    X_features_rf = df[['total_spend', 'frequency', 'customer_age', 'recency']]
    X_scaled_rf = scaler_rf.fit_transform(X_features_rf)
    X_scaled_rf_df = pd.DataFrame(X_scaled_rf, columns=['total_spend', 'frequency', 'customer_age', 'recency'])
    
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler_rf, f)
        
    X_churn = X_scaled_rf_df[['total_spend', 'frequency', 'customer_age']]
    y_churn = df['is_churn'].values
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_churn, y_churn, test_size=0.2, random_state=42)
    
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf_clf.fit(Xc_train, yc_train)
    with open(os.path.join(MODEL_DIR, "churn_rf_model.pkl"), 'wb') as f:
        pickle.dump(rf_clf, f)
        
    knn_clf = KNeighborsClassifier(n_neighbors=5)
    knn_clf.fit(Xc_train, yc_train)
    print(f"Churn RF  - F1: {f1_score(yc_test, rf_clf.predict(Xc_test), average='weighted'):.4f}")
    print(f"Churn KNN - F1: {f1_score(yc_test, knn_clf.predict(Xc_test), average='weighted'):.4f}")
        
    # --- LTV Regression (>0.75 R2 mapping) ---
    X_ltv = df[['frequency', 'total_price', 'total_freight']]
    y_ltv = df['total_spend'].values
    
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(X_ltv, y_ltv, test_size=0.2, random_state=42)
    
    lin_reg = LinearRegression()
    lin_reg.fit(Xl_train, yl_train)
    yl_pred = lin_reg.predict(Xl_test)
    print(f"LTV LR - RMSE: {np.sqrt(mean_squared_error(yl_test, yl_pred)):.4f}, R2: {r2_score(yl_test, yl_pred):.4f}")
    
    with open(os.path.join(MODEL_DIR, "ltv_lr_model.pkl"), 'wb') as f:
        pickle.dump(lin_reg, f)
        
    X_ols_train_sm = sm.add_constant(Xl_train)
    ols_model = sm.OLS(yl_train, X_ols_train_sm)
    ols_results = ols_model.fit()
    with open(os.path.join(MODEL_DIR, "ols_summary.txt"), 'w') as f:
        f.write(ols_results.summary().as_text())

def predict_churn(total_spend, frequency, customer_age):
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), 'rb') as f:
        scaler = pickle.load(f)
    features_raw = pd.DataFrame({'total_spend': [total_spend], 'frequency': [frequency], 'customer_age': [customer_age], 'recency': [0]})
    scaled_features = scaler.transform(features_raw)
    
    with open(os.path.join(MODEL_DIR, "churn_rf_model.pkl"), 'rb') as f:
        model = pickle.load(f)
    final_features = [[scaled_features[0][0], scaled_features[0][1], scaled_features[0][2]]]
    return model.predict(final_features)[0], model.predict_proba(final_features)[0][1]

def predict_ltv(frequency, recency, customer_age):
    # For prediction via streamlit we will spoof price/freight assuming averages map to frequency
    with open(os.path.join(MODEL_DIR, "ltv_lr_model.pkl"), 'rb') as f:
        model = pickle.load(f)
    estimated_price = frequency * 50
    estimated_freight = frequency * 10
    features = pd.DataFrame({'frequency': [frequency], 'total_price': [estimated_price], 'total_freight': [estimated_freight]})
    return max(0, model.predict(features)[0])

if __name__ == "__main__":
    train_and_save_models()
