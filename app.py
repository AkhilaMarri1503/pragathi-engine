"""
Pragathi Dynamic Predictive & Inventory Optimization Engine
Simplified - No external auth dependencies
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import re
import warnings
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Visualization
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pragathi Analysis Engine",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = True  # Skip auth for simplicity
if 'data' not in st.session_state:
    st.session_state.data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'inventory_results' not in st.session_state:
    st.session_state.inventory_results = None

def auto_detect_target(df):
    """Auto-detect target column"""
    # Exclude ID-like columns
    exclude_patterns = ['id', 'index', 'key', 'row', 'date', 'time', 'stock', 'sku']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(p in col_lower for p in exclude_patterns):
            candidates.append(col)
    
    # Score candidates
    target_keywords = ['sales', 'demand', 'revenue', 'quantity', 'qty', 'price', 'amount', 'value']
    
    best_col = None
    best_score = 0
    
    for col in candidates:
        score = 0
        col_lower = col.lower()
        for keyword in target_keywords:
            if keyword in col_lower:
                score += 10
        if score > best_score:
            best_score = score
            best_col = col
    
    if best_col:
        confidence = "High 🟢" if best_score >= 10 else "Medium 🟡" if best_score >= 5 else "Low 🟠"
        return best_col, confidence
    return None, "None 🔴"

def clean_data(df):
    """Basic data cleaning"""
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    df = df.drop_duplicates()
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().any():
            df[col].fillna("Unknown", inplace=True)
    
    return df

def auto_detect_inventory_columns(df):
    """Detect inventory columns"""
    detected = {'stock_col': None, 'lead_time_col': None, 'sku_col': None}
    
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['stock', 'inventory', 'on_hand']):
            detected['stock_col'] = col
        if any(x in col_lower for x in ['lead', 'delivery']):
            detected['lead_time_col'] = col
        if any(x in col_lower for x in ['sku', 'product', 'item']):
            detected['sku_col'] = col
    
    return detected

def train_model(df, target_col):
    """Train Random Forest model"""
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(0)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Predict
    predictions = model.predict(X)
    df[f'Predicted_{target_col}'] = predictions
    
    # Calculate metrics
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    
    return df, model, mae, r2

def calculate_inventory(df, target_col, inv_cols, z_score=1.65):
    """Calculate inventory optimization"""
    df = df.copy()
    
    stock_col = inv_cols.get('stock_col')
    if not stock_col:
        return df
    
    avg_demand = df[target_col].mean()
    demand_std = df[target_col].std()
    
    df['Safety_Stock'] = z_score * demand_std
    df['Reorder_Point'] = avg_demand + df['Safety_Stock']
    
    df['Procurement_Status'] = df.apply(
        lambda row: "⚠️ TRIGGER PURCHASE ORDER" if row[stock_col] < row['Reorder_Point'] else "✅ STOCK SUFFICIENT",
        axis=1
    )
    
    df['Recommended_Order_Qty'] = df.apply(
        lambda row: max(0, (avg_demand * 4) + row['Safety_Stock'] - row[stock_col]) 
        if row['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER" else 0,
        axis=1
    )
    
    return df

def main():
    """Main app"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Pragathi Dynamic Predictive & Inventory Optimization Engine</h1>
        <p>AI-Powered Supply Chain Analytics for Smarter Decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload
    uploaded_file = st.file_uploader(
        "📁 Drag and drop your CSV or Excel file here",
        type=['csv', 'xlsx']
    )
    
    if uploaded_file:
        # Load file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean data
        with st.spinner("🧹 Cleaning data..."):
            df = clean_data(df)
            st.session_state.cleaned_data = df
        
        # Show preview
        with st.expander("👁️ Data Preview"):
            st.dataframe(df.head(10))
        
        # Target detection
        detected_target, confidence = auto_detect_target(df)
        
        if detected_target:
            st.success(f"🎯 Auto-detected target: **{detected_target}** (Confidence: {confidence})")
            target_col = st.selectbox(
                "Or select a different target column:",
                df.select_dtypes(include=[np.number]).columns,
                index=list(df.select_dtypes(include=[np.number]).columns).index(detected_target) 
                if detected_target in df.columns else 0
            )
        else:
            target_col = st.selectbox(
                "Select your target column (what you want to predict):",
                df.select_dtypes(include=[np.number]).columns
            )
        
        st.session_state.target_column = target_col
        
        # Run analysis button
        if st.button("🚀 Run Predictive Optimization", use_container_width=True):
            with st.spinner("Training models and analyzing data..."):
                # Train model
                df, model, mae, r2 = train_model(df, target_col)
                st.session_state.predictions = df
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
                with col2:
                    st.metric("R² Score", f"{r2:.3f}")
                
                # Actual vs Predicted plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df[target_col],
                    y=df[f'Predicted_{target_col}'],
                    mode='markers',
                    name='Predictions',
                    marker=dict(size=8, color='#4CAF50')
                ))
                
                # Perfect line
                min_val = min(df[target_col].min(), df[f'Predicted_{target_col}'].min())
                max_val = max(df[target_col].max(), df[f'Predicted_{target_col}'].max())
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="Actual vs Predicted Values",
                    xaxis_title=f"Actual {target_col}",
                    yaxis_title=f"Predicted {target_col}",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Inventory optimization
                inv_cols = auto_detect_inventory_columns(df)
                
                if inv_cols['stock_col']:
                    st.markdown("### 📦 Inventory Optimization")
                    
                    df_inv = calculate_inventory(df, target_col, inv_cols)
                    st.session_state.inventory_results = df_inv
                    
                    # Summary
                    trigger_count = (df_inv['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER").sum()
                    total_order = df_inv['Recommended_Order_Qty'].sum()
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Items Needing Order", trigger_count)
                    with col2:
                        st.metric("Total Order Quantity", f"{int(total_order):,}")
                    with col3:
                        st.metric("Avg Safety Stock", f"{df_inv['Safety_Stock'].mean():.0f}")
                    
                    # Show table
                    display_cols = ['Procurement_Status', 'Safety_Stock', 'Reorder_Point', 'Recommended_Order_Qty']
                    if inv_cols['sku_col']:
                        display_cols = [inv_cols['sku_col']] + display_cols
                    
                    st.dataframe(df_inv[display_cols].head(20), use_container_width=True)
                else:
                    st.info("💡 Tip: Add 'Stock' or 'Inventory' column for inventory optimization")
                
                # Explanations
                st.markdown("### 📋 What This Means For You")
                
                explanation = f"""
                <div class="success-card">
                <strong>🤖 Your best prediction model is: Random Forest</strong><br>
                📊 On average, predictions are off by <strong>{mae:.2f}</strong> units.<br>
                📈 The model explains <strong>{r2*100:.1f}%</strong> of the patterns in your data.<br>
                </div>
                """
                st.markdown(explanation, unsafe_allow_html=True)
                
                # Download button
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    "📥 Download Complete Results (CSV)",
                    csv,
                    f"optimized_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
                st.success("✅ Analysis complete!")
    
    else:
        st.info("👈 Please upload your CSV or Excel file to begin.")

if __name__ == "__main__":
    main()
