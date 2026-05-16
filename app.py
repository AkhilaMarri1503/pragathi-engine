"""
Pragathi Dynamic Predictive & Inventory Optimization Engine
With Pie Charts and Bar Graphs for Better Visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import re
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
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
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = True
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
    original_rows = len(df)
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
        if any(x in col_lower for x in ['stock', 'inventory', 'on_hand', 'current_stock']):
            detected['stock_col'] = col
        if any(x in col_lower for x in ['lead', 'delivery', 'lead_time']):
            detected['lead_time_col'] = col
        if any(x in col_lower for x in ['sku', 'product', 'item', 'product_id']):
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
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predict
    predictions = model.predict(X)
    df[f'Predicted_{target_col}'] = predictions
    
    # Calculate metrics
    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
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
        type=['csv', 'xlsx'],
        help="Upload your sales, demand, or inventory data"
    )
    
    if uploaded_file:
        # Load file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File loaded: {len(df):,} rows, {len(df.columns)} columns")
            
            # Clean data
            with st.spinner("🧹 Cleaning data..."):
                df = clean_data(df)
                st.session_state.cleaned_data = df
            
            # Show preview
            with st.expander("👁️ Data Preview (First 5 rows)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)
            
            # Show column info
            st.markdown("### 📊 Column Types")
            col_types = []
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    col_types.append({"Column": col, "Type": "🟢 Numeric"})
                else:
                    col_types.append({"Column": col, "Type": "🟡 Text/Categorical"})
            st.dataframe(pd.DataFrame(col_types), use_container_width=True)
            
            # Target detection
            st.markdown("### 🎯 Target Selection")
            
            detected_target, confidence = auto_detect_target(df)
            
            if detected_target:
                st.success(f"🎯 Auto-detected target: **{detected_target}** (Confidence: {confidence})")
                
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                default_index = numeric_cols.index(detected_target) if detected_target in numeric_cols else 0
                
                target_col = st.selectbox(
                    "Select your target variable (what you want to predict):",
                    numeric_cols,
                    index=default_index
                )
            else:
                target_col = st.selectbox(
                    "Select your target variable (what you want to predict):",
                    df.select_dtypes(include=[np.number]).columns
                )
            
            st.session_state.target_column = target_col
            
            # Run analysis button
            if st.button("🚀 Run Predictive Optimization", use_container_width=True):
                with st.spinner("🔄 Training models and analyzing data... This may take a moment."):
                    try:
                        # Train model
                        df_result, model, mae, r2 = train_model(df, target_col)
                        st.session_state.predictions = df_result
                        
                        # Display metrics
                        st.markdown("### 📈 Model Performance")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📉 Mean Absolute Error (MAE)", f"{mae:.2f}")
                        with col2:
                            st.metric("📈 R² Score", f"{r2:.3f}")
                        
                        # ============ BAR CHARTS FOR PREDICTIONS ============
                        st.markdown("### 📊 Prediction Distribution")
                        
                        # Bar Chart 1: Actual vs Predicted Averages
                        comparison_df = pd.DataFrame({
                            'Metric': ['Actual Average', 'Predicted Average'],
                            'Value': [df_result[target_col].mean(), df_result[f'Predicted_{target_col}'].mean()]
                        })
                        
                        fig1 = px.bar(
                            comparison_df, 
                            x='Metric', 
                            y='Value',
                            title=f'Actual vs Predicted {target_col} (Average Values)',
                            text='Value',
                            color='Metric',
                            color_discrete_map={'Actual Average': '#4CAF50', 'Predicted Average': '#FF6B6B'}
                        )
                        fig1.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig1.update_layout(height=450, showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                        
                        # Bar Chart 2: Top 10 Predictions
                        st.markdown("### 📈 Top 10 Highest Predictions")
                        top_predictions = df_result.nlargest(10, f'Predicted_{target_col}')[[target_col, f'Predicted_{target_col}']]
                        top_predictions = top_predictions.reset_index()
                        
                        fig2 = px.bar(
                            top_predictions,
                            x=top_predictions.index,
                            y=f'Predicted_{target_col}',
                            title=f'Top 10 Predicted {target_col} Values',
                            labels={'x': 'Row Number', f'Predicted_{target_col}': f'Predicted {target_col}'},
                            color=f'Predicted_{target_col}',
                            color_continuous_scale='Viridis',
                            text=f'Predicted_{target_col}'
                        )
                        fig2.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                        fig2.update_layout(height=450)
                        st.plotly_chart(fig2, use_container_width=True)
                        
                        # Bar Chart 3: Distribution of Predictions
                        st.markdown("### 📊 Prediction Value Distribution")
                        fig3 = px.histogram(
                            df_result,
                            x=f'Predicted_{target_col}',
                            nbins=20,
                            title=f'Distribution of Predicted {target_col} Values',
                            color_discrete_sequence=['#667eea'],
                            labels={f'Predicted_{target_col}': f'Predicted {target_col}', 'count': 'Frequency'}
                        )
                        fig3.update_layout(height=400)
                        st.plotly_chart(fig3, use_container_width=True)
                        
                        # Show prediction table
                        with st.expander("📋 View Predictions Table", expanded=False):
                            display_df = df_result[[target_col, f'Predicted_{target_col}']].head(20)
                            st.dataframe(display_df, use_container_width=True)
                        
                        # ============ INVENTORY OPTIMIZATION ============
                        st.markdown("### 📦 Inventory Optimization")
                        
                        inv_cols = auto_detect_inventory_columns(df_result)
                        
                        if inv_cols['stock_col']:
                            df_inv = calculate_inventory(df_result, target_col, inv_cols)
                            st.session_state.inventory_results = df_inv
                            
                            # Summary metrics
                            trigger_count = (df_inv['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER").sum()
                            sufficient_count = (df_inv['Procurement_Status'] == "✅ STOCK SUFFICIENT").sum()
                            total_order = df_inv['Recommended_Order_Qty'].sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🚨 Items Needing Order", trigger_count, delta=f"{(trigger_count/len(df_inv)*100):.1f}%")
                            with col2:
                                st.metric("✅ Items Sufficient", sufficient_count)
                            with col3:
                                st.metric("📦 Total Order Quantity", f"{int(total_order):,}")
                            
                            # ============ PIE CHART - Procurement Status ============
                            st.markdown("### 🥧 Procurement Status Breakdown")
                            
                            procurement_counts = df_inv['Procurement_Status'].value_counts()
                            fig_pie = px.pie(
                                values=procurement_counts.values,
                                names=procurement_counts.index,
                                title='Items Status Distribution',
                                color_discrete_map={'⚠️ TRIGGER PURCHASE ORDER': '#FF6B6B', '✅ STOCK SUFFICIENT': '#4CAF50'},
                                hole=0.3
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            fig_pie.update_layout(height=450)
                            st.plotly_chart(fig_pie, use_container_width=True)
                            
                            # ============ BAR CHART - Top Items to Order ============
                            st.markdown("### 📊 Top 10 Items to Order")
                            
                            items_to_order = df_inv[df_inv['Recommended_Order_Qty'] > 0].nlargest(10, 'Recommended_Order_Qty')
                            
                            if len(items_to_order) > 0:
                                # Use SKU column if available, otherwise use index
                                if inv_cols['sku_col']:
                                    x_axis = inv_cols['sku_col']
                                    items_to_order_display = items_to_order[[x_axis, 'Recommended_Order_Qty', 'Safety_Stock']]
                                else:
                                    items_to_order_display = items_to_order.reset_index()
                                    x_axis = 'index'
                                    items_to_order_display = items_to_order_display.rename(columns={'index': 'Item Number'})
                                    x_axis = 'Item Number'
                                
                                fig_bar = px.bar(
                                    items_to_order_display,
                                    x=x_axis,
                                    y='Recommended_Order_Qty',
                                    title='Top 10 Items Requiring Purchase Order',
                                    labels={'Recommended_Order_Qty': 'Recommended Order Quantity', x_axis: 'Item'},
                                    color='Recommended_Order_Qty',
                                    color_continuous_scale='Reds',
                                    text='Recommended_Order_Qty'
                                )
                                fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                                fig_bar.update_layout(height=450)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("✅ No items need ordering at this time!")
                            
                            # Show procurement table with highlighting
                            st.markdown("#### 📋 Detailed Procurement Plan")
                            display_cols = ['Procurement_Status', 'Safety_Stock', 'Reorder_Point', 'Recommended_Order_Qty']
                            if inv_cols['sku_col']:
                                display_cols = [inv_cols['sku_col']] + display_cols
                            
                            # Add highlighting
                            def highlight_trigger(row):
                                if row['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER":
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)
                            
                            styled_df = df_inv[display_cols].head(20).style.apply(highlight_trigger, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Alert if items need ordering
                            if trigger_count > 0:
                                st.warning(f"⚠️ **URGENT:** {trigger_count} items need immediate reordering! Check the table above for details.")
                            else:
                                st.success("✅ All items have sufficient stock! No immediate action needed.")
                        else:
                            st.info("💡 **Tip:** Add a column named 'Stock', 'Inventory', or 'Current_Stock' to enable inventory optimization calculations.")
                        
                        # ============ SIMPLE EXPLANATIONS ============
                        st.markdown("### 📋 What This Means For You")
                        
                        if r2 > 0.7:
                            quality_text = "excellent"
                            quality_emoji = "🌟"
                        elif r2 > 0.5:
                            quality_text = "good"
                            quality_emoji = "👍"
                        else:
                            quality_text = "moderate"
                            quality_emoji = "📊"
                        
                        explanation = f"""
                        <div class="success-card">
                        <strong>🤖 Model Summary:</strong><br>
                        • Best model: <strong>Random Forest</strong><br>
                        • Average prediction error: <strong>{mae:.2f}</strong> units<br>
                        • Model explains <strong>{r2*100:.1f}%</strong> of the patterns in your data ({quality_text} {quality_emoji})<br>
                        </div>
                        """
                        st.markdown(explanation, unsafe_allow_html=True)
                        
                        if inv_cols.get('stock_col') and trigger_count > 0:
                            st.markdown(f"""
                            <div class="warning-card">
                            <strong>📦 Inventory Recommendations:</strong><br>
                            • <strong>{trigger_count}</strong> out of <strong>{len(df_inv)}</strong> items need reordering<br>
                            • Total order quantity: <strong>{int(total_order):,}</strong> units<br>
                            • Recommended safety stock level: <strong>{df_inv['Safety_Stock'].mean():.0f}</strong> units average<br>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # ============ DOWNLOAD BUTTON ============
                        st.markdown("### 📥 Export Results")
                        
                        csv_buffer = io.StringIO()
                        df_result.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue().encode()
                        
                        st.download_button(
                            "📥 Download Complete Results (CSV)",
                            csv_data,
                            f"optimized_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        st.balloons()
                        st.success("✅ Analysis complete! Your results are ready to download.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
                        st.info("Please try selecting a different target column or check your data format.")
        
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV or Excel file.")
    
    else:
        # Welcome message
        st.info("👈 **Get Started:** Upload your CSV or Excel file to begin the analysis.")
        
        with st.expander("ℹ️ How to use this app", expanded=True):
            st.markdown("""
            ### 📋 Instructions:
            
            1. **Upload your data** - CSV or Excel file with your sales/demand data
            2. **Select target column** - Choose what you want to predict (e.g., Sales, Quantity)
            3. **Click Run** - The AI will analyze and optimize
            
            ### 📊 What you need:
            - At least 10 rows of data
            - A numeric column to predict (like sales or demand)
            - Optional: Stock/Inventory column for procurement recommendations
            
            ### 🎯 Features:
            - Automatic data cleaning
            - Demand prediction using Random Forest
            - **Pie charts** for procurement status breakdown
            - **Bar graphs** for top items to order
            - Safety stock calculation
            - Reorder point recommendations
            - CSV export of results
            """)
        
        # Example data preview
        st.markdown("### 📋 Example Data Format")
        example_df = pd.DataFrame({
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Product': ['Product A', 'Product B', 'Product A'],
            'Sales': [100, 150, 120],
            'Current_Stock': [500, 300, 480],
            'Lead_Time': [5, 7, 5]
        })
        st.dataframe(example_df, use_container_width=True)
        st.caption("💡 Your file should have similar columns. 'Sales' would be your target column.")

if __name__ == "__main__":
    main()
