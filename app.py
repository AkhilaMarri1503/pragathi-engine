"""
Simplified Pragathi Dynamic Predictive & Inventory Optimization Engine
No PyCaret dependency - uses scikit-learn directly
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pragathi Predictive Engine",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border-radius: 8px;
    }
    .highlight-danger {
        background-color: #ffcccc;
    }
</style>
""", unsafe_allow_html=True)

class DataCleaningEngine:
    @staticmethod
    def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            df.columns = df.columns.str.strip()
            
            # Date conversion
            date_patterns = ['date', 'time', 'transaction', 'datetime']
            for col in df.select_dtypes(include=['object']).columns:
                if any(pattern in col.lower() for pattern in date_patterns):
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        df[f'{col}_Month'] = df[col].dt.month
                        df[f'{col}_Week'] = df[col].dt.isocalendar().week
                    except:
                        pass
            
            # Handle missing values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if df[col].isnull().any():
                    df[col].fillna("Unknown", inplace=True)
            
            df = df.dropna()
            st.success(f"✅ Data cleaned: {df.shape}")
            return df
        except Exception as e:
            st.error(f"Error: {str(e)}")
            return df

class SimpleMLEngine:
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.copy()
        self.target_col = target_col
        self.model = None
        
    def prepare_features(self):
        df_ml = self.df.copy()
        
        # Encode categorical variables
        for col in df_ml.select_dtypes(include=['object']).columns:
            if col != self.target_col:
                df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))
        
        # Remove non-numeric columns that can't be used
        exclude_cols = []
        for col in df_ml.columns:
            if col != self.target_col:
                if not pd.api.types.is_numeric_dtype(df_ml[col]):
                    exclude_cols.append(col)
        
        if exclude_cols:
            df_ml = df_ml.drop(columns=exclude_cols)
            st.info(f"Excluded non-numeric columns: {exclude_cols[:3]}")
        
        return df_ml
    
    def train_and_predict(self):
        try:
            df_ml = self.prepare_features()
            
            # Prepare features and target
            feature_cols = [col for col in df_ml.columns if col != self.target_col]
            X = df_ml[feature_cols]
            y = df_ml[self.target_col]
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.model.fit(X_train, y_train)
            
            # Make predictions
            predictions = self.model.predict(X)
            self.df['Predicted_Demand'] = predictions
            
            # Calculate metrics
            y_pred_test = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            r2 = r2_score(y_test, y_pred_test)
            
            metrics = {
                'model_name': 'Random Forest',
                'mae': mae,
                'r2': r2
            }
            
            st.success(f"✅ Model trained! MAE: {mae:.2f}, R²: {r2:.3f}")
            return self.df, metrics
            
        except Exception as e:
            st.error(f"ML Error: {str(e)}")
            # Fallback to simple moving average
            self.df['Predicted_Demand'] = self.df[self.target_col].rolling(3, min_periods=1).mean()
            return self.df, {'model_name': 'Baseline', 'mae': None, 'r2': None}

class InventoryOptimizer:
    def __init__(self, df: pd.DataFrame, demand_col: str):
        self.df = df.copy()
        self.demand_col = demand_col
        
    def optimize_inventory(self) -> pd.DataFrame:
        try:
            # Auto-detect columns
            stock_col = None
            lead_col = None
            
            for col in self.df.columns:
                col_lower = col.lower()
                if 'stock' in col_lower or 'inventory' in col_lower:
                    stock_col = col
                if 'lead' in col_lower or 'delivery' in col_lower:
                    lead_col = col
            
            if not stock_col:
                st.warning("No inventory column detected")
                return self.df
            
            # Calculate metrics
            avg_demand = self.df[self.demand_col].mean()
            demand_std = self.df[self.demand_col].std()
            lead_weeks = 1.0  # Default
            
            if lead_col:
                lead_weeks = self.df[lead_col].mean() / 7.0
            
            # Safety Stock (95% service level, Z=1.65)
            self.df['Safety_Stock'] = 1.65 * np.sqrt(lead_weeks * (demand_std ** 2))
            
            # Reorder Point
            self.df['Reorder_Point'] = (avg_demand * lead_weeks) + self.df['Safety_Stock']
            
            # Procurement decisions
            self.df['Procurement_Status'] = self.df.apply(
                lambda row: "⚠️ TRIGGER PURCHASE ORDER" 
                if row[stock_col] < row['Reorder_Point']
                else "✅ STOCK SUFFICIENT",
                axis=1
            )
            
            # Order quantity (4 weeks coverage)
            self.df['Recommended_Order_Qty'] = self.df.apply(
                lambda row: max(0, (avg_demand * 4) + row['Safety_Stock'] - row[stock_col])
                if row['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER"
                else 0,
                axis=1
            )
            
            return self.df
        except Exception as e:
            st.error(f"Inventory error: {str(e)}")
            return self.df

def main():
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; margin-bottom: 30px;">
        <h1 style="color: white;">📈 Pragathi Predictive & Inventory Optimization Engine</h1>
        <p style="color: white;">AI-Powered Supply Chain Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Clean data
            with st.spinner("Cleaning data..."):
                cleaned_df = DataCleaningEngine.sanitize_dataframe(df)
            
            with st.expander("Data Preview"):
                st.dataframe(cleaned_df.head())
            
            target_column = st.selectbox("Select target variable to predict", cleaned_df.columns)
            
            if st.button("Run Optimization", type="primary"):
                with st.spinner("Processing..."):
                    tab1, tab2 = st.tabs(["📊 Predictions", "📦 Inventory"])
                    
                    with tab1:
                        ml_engine = SimpleMLEngine(cleaned_df, target_column)
                        predicted_df, metrics = ml_engine.train_and_predict()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Model", metrics['model_name'])
                        with col2:
                            if metrics['mae']:
                                st.metric("MAE", f"{metrics['mae']:.2f}")
                        with col3:
                            if metrics['r2']:
                                st.metric("R²", f"{metrics['r2']:.3f}")
                        
                        st.dataframe(predicted_df[[target_column, 'Predicted_Demand']].head(20))
                    
                    with tab2:
                        optimizer = InventoryOptimizer(predicted_df, target_column)
                        final_df = optimizer.optimize_inventory()
                        
                        if 'Procurement_Status' in final_df.columns:
                            trigger_count = (final_df['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER").sum()
                            st.metric("Items Needing Order", trigger_count)
                            
                            st.dataframe(final_df[[target_column, 'Predicted_Demand', 'Safety_Stock', 
                                                  'Reorder_Point', 'Procurement_Status', 'Recommended_Order_Qty']])
                            
                            # Download button
                            csv = final_df.to_csv(index=False)
                            st.download_button("Download Plan", csv, "optimized_plan.csv", "text/csv")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()