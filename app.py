"""
Pragathi Dynamic Predictive & Inventory Optimization Engine
Simplified with Optional Charts and Reason Analysis
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
    .info-card {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
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
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'inventory_results' not in st.session_state:
    st.session_state.inventory_results = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

def auto_detect_target(df):
    """Auto-detect target column"""
    exclude_patterns = ['id', 'index', 'key', 'row', 'date', 'time', 'stock', 'sku']
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        if not any(p in col_lower for p in exclude_patterns):
            candidates.append(col)
    
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
    detected = {'stock_col': None, 'lead_time_col': None, 'sku_col': None, 'article_col': None}
    
    for col in df.columns:
        col_lower = col.lower()
        if any(x in col_lower for x in ['stock', 'inventory', 'on_hand', 'current_stock']):
            detected['stock_col'] = col
        if any(x in col_lower for x in ['lead', 'delivery', 'lead_time']):
            detected['lead_time_col'] = col
        if any(x in col_lower for x in ['sku', 'product', 'item']):
            detected['sku_col'] = col
        if any(x in col_lower for x in ['article', 'artikel', 'art_no']):
            detected['article_col'] = col
    
    return detected

def analyze_reasons(df, target_col, date_col=None):
    """Analyze reasons for increase/decrease in sales/inventory"""
    reasons = {}
    
    # Calculate overall trend
    if len(df) > 1 and target_col in df.columns:
        first_value = df[target_col].iloc[0]
        last_value = df[target_col].iloc[-1]
        
        if last_value > first_value:
            percent_change = ((last_value - first_value) / first_value) * 100
            reasons['trend'] = f"📈 **Increasing** by {percent_change:.1f}%"
            reasons['trend_explanation'] = f"Your {target_col} is growing. This could be due to increased demand, successful marketing, or seasonal factors."
        elif last_value < first_value:
            percent_change = ((first_value - last_value) / first_value) * 100
            reasons['trend'] = f"📉 **Decreasing** by {percent_change:.1f}%"
            reasons['trend_explanation'] = f"Your {target_col} is declining. Consider checking inventory levels, pricing, or market conditions."
        else:
            reasons['trend'] = f"➡️ **Stable**"
            reasons['trend_explanation'] = f"Your {target_col} remains stable, indicating consistent demand patterns."
    
    # Analyze peak periods
    if date_col and date_col in df.columns:
        df['temp_date'] = pd.to_datetime(df[date_col])
        df['month'] = df['temp_date'].dt.month
        monthly_avg = df.groupby('month')[target_col].mean()
        best_month = monthly_avg.idxmax()
        reasons['best_month'] = f"📅 Best month: Month {best_month} with average {monthly_avg.max():.0f} units"
        reasons['best_month_explanation'] = f"Month {best_month} shows highest {target_col}. Plan inventory accordingly."
    
    # Analyze outliers
    mean_val = df[target_col].mean()
    std_val = df[target_col].std()
    high_outliers = df[df[target_col] > mean_val + 2*std_val]
    if len(high_outliers) > 0:
        reasons['outliers'] = f"⚠️ Found {len(high_outliers)} unusually high values"
        reasons['outliers_explanation'] = "These could be promotional periods, holidays, or data entry errors."
    
    return reasons

def train_model(df, target_col):
    """Train Random Forest model"""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X)
    df[f'Predicted_{target_col}'] = predictions
    
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
            if st.button("✅ Run Predictive Optimization", use_container_width=True):
                with st.spinner("🔄 Analyzing your data..."):
                    try:
                        # Train model
                        df_result, model, mae, r2 = train_model(df, target_col)
                        st.session_state.predictions = df_result
                        
                        # Detect inventory columns
                        inv_cols = auto_detect_inventory_columns(df_result)
                        
                        # Calculate inventory if stock column exists
                        if inv_cols['stock_col']:
                            df_inv = calculate_inventory(df_result, target_col, inv_cols)
                            st.session_state.inventory_results = df_inv
                        else:
                            df_inv = df_result
                            st.session_state.inventory_results = df_result
                        
                        st.session_state.analysis_complete = True
                        
                        # ============ COMPLETION MESSAGE ============
                        st.success("✅ Analysis Complete!")
                        
                        # ============ PRODUCT/ARTICLE INFORMATION ============
                        st.markdown("### 📦 Product & Article Information")
                        
                        if inv_cols.get('sku_col') or inv_cols.get('article_col'):
                            product_col = inv_cols.get('sku_col') or inv_cols.get('article_col')
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**🔑 Product/Article Column:** `{product_col}`")
                                st.markdown(f"**📊 Total Products:** {df_inv[product_col].nunique()}")
                            with col2:
                                unique_products = df_inv[product_col].unique()[:10]
                                st.markdown("**🏷️ Sample Products/Articles:**")
                                st.write(", ".join([str(p) for p in unique_products[:5]]))
                        else:
                            st.info("ℹ️ No product/article column detected. Add columns named 'SKU', 'Product', or 'Article' for product-level insights.")
                        
                        # ============ PREDICTION RESULTS ============
                        st.markdown("### 📊 Prediction Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("📉 Mean Absolute Error (MAE)", f"{mae:.2f}")
                        with col2:
                            st.metric("📈 R² Score (Accuracy)", f"{r2:.3f}")
                        
                        # Show prediction table with product info
                        st.markdown("#### 🔮 Predicted Values")
                        
                        display_cols = [target_col, f'Predicted_{target_col}']
                        if inv_cols.get('sku_col'):
                            display_cols = [inv_cols['sku_col']] + display_cols
                        elif inv_cols.get('article_col'):
                            display_cols = [inv_cols['article_col']] + display_cols
                        
                        st.dataframe(df_inv[display_cols].head(10), use_container_width=True)
                        
                        # ============ REASON ANALYSIS ============
                        st.markdown("### 🔍 Reason Analysis")
                        
                        # Detect date column
                        date_col = None
                        for col in df.columns:
                            if 'date' in col.lower() or 'time' in col.lower():
                                date_col = col
                                break
                        
                        reasons = analyze_reasons(df_inv, target_col, date_col)
                        
                        # Display reasons
                        if 'trend' in reasons:
                            st.markdown(f"**{reasons['trend']}**")
                            st.markdown(f"💡 {reasons['trend_explanation']}")
                        
                        if 'best_month' in reasons:
                            st.markdown(f"**{reasons['best_month']}**")
                            st.markdown(f"💡 {reasons['best_month_explanation']}")
                        
                        if 'outliers' in reasons:
                            st.markdown(f"**{reasons['outliers']}**")
                            st.markdown(f"💡 {reasons['outliers_explanation']}")
                        
                        # ============ OPTIONAL CHARTS (User Selects) ============
                        st.markdown("### 📈 Optional Charts")
                        st.info("💡 Select the charts you want to view below:")
                        
                        show_pie = st.checkbox("🥧 Show Procurement Status Pie Chart", value=False)
                        show_bar = st.checkbox("📊 Show Top Items to Order (Bar Graph)", value=False)
                        show_hist = st.checkbox("📉 Show Prediction Distribution (Histogram)", value=False)
                        
                        # Pie Chart
                        if show_pie and inv_cols.get('stock_col'):
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
                        
                        # Bar Chart - Top Items to Order
                        if show_bar and inv_cols.get('stock_col'):
                            items_to_order = df_inv[df_inv['Recommended_Order_Qty'] > 0].nlargest(10, 'Recommended_Order_Qty')
                            
                            if len(items_to_order) > 0:
                                if inv_cols.get('sku_col'):
                                    x_axis = inv_cols['sku_col']
                                elif inv_cols.get('article_col'):
                                    x_axis = inv_cols['article_col']
                                else:
                                    items_to_order = items_to_order.reset_index()
                                    x_axis = 'index'
                                
                                fig_bar = px.bar(
                                    items_to_order,
                                    x=x_axis,
                                    y='Recommended_Order_Qty',
                                    title='Top 10 Items Requiring Purchase Order',
                                    labels={'Recommended_Order_Qty': 'Recommended Order Quantity', x_axis: 'Product/Article'},
                                    color='Recommended_Order_Qty',
                                    color_continuous_scale='Reds',
                                    text='Recommended_Order_Qty'
                                )
                                fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                                fig_bar.update_layout(height=450)
                                st.plotly_chart(fig_bar, use_container_width=True)
                            else:
                                st.info("✅ No items need ordering - no bar chart to display")
                        
                        # Histogram
                        if show_hist:
                            fig_hist = px.histogram(
                                df_inv,
                                x=f'Predicted_{target_col}',
                                nbins=20,
                                title=f'Distribution of Predicted {target_col} Values',
                                color_discrete_sequence=['#667eea'],
                                labels={f'Predicted_{target_col}': f'Predicted {target_col}', 'count': 'Frequency'}
                            )
                            fig_hist.update_layout(height=450)
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # ============ INVENTORY SUMMARY ============
                        if inv_cols.get('stock_col'):
                            st.markdown("### 📦 Inventory Summary")
                            
                            trigger_count = (df_inv['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER").sum()
                            sufficient_count = (df_inv['Procurement_Status'] == "✅ STOCK SUFFICIENT").sum()
                            total_order = df_inv['Recommended_Order_Qty'].sum()
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🚨 Items Needing Order", trigger_count)
                            with col2:
                                st.metric("✅ Items Sufficient", sufficient_count)
                            with col3:
                                st.metric("📦 Total Order Quantity", f"{int(total_order):,}")
                            
                            # Detailed procurement table with highlighting
                            st.markdown("#### 📋 Detailed Procurement Plan")
                            
                            def highlight_trigger(row):
                                if row['Procurement_Status'] == "⚠️ TRIGGER PURCHASE ORDER":
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)
                            
                            display_cols = ['Procurement_Status', 'Safety_Stock', 'Reorder_Point', 'Recommended_Order_Qty']
                            if inv_cols.get('sku_col'):
                                display_cols = [inv_cols['sku_col']] + display_cols
                            elif inv_cols.get('article_col'):
                                display_cols = [inv_cols['article_col']] + display_cols
                            
                            styled_df = df_inv[display_cols].head(20).style.apply(highlight_trigger, axis=1)
                            st.dataframe(styled_df, use_container_width=True)
                            
                            if trigger_count > 0:
                                st.warning(f"⚠️ **Action Required:** {trigger_count} items need immediate reordering!")
                            else:
                                st.success("✅ All items have sufficient stock!")
                        
                        # ============ DOWNLOAD ============
                        st.markdown("### 📥 Download Results")
                        
                        csv_buffer = io.StringIO()
                        df_inv.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue().encode()
                        
                        st.download_button(
                            "📥 Download Complete Results (CSV)",
                            csv_data,
                            f"optimized_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                        
                        st.info("✅ Analysis complete! Use the checkboxes above to view additional charts.")
                        
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
            
            ### 📊 What you'll see:
            - **Product/Article numbers** with predictions
            - **Reasons for increase/decrease** in sales
            - **Predicted values** for each product
            - **Optional charts** (select what you want to see)
            - **Procurement recommendations**
            
            ### 🎯 Features:
            - Automatic data cleaning
            - Sales/Inventory prediction
            - Trend analysis with explanations
            - Product/Article level insights
            - Safety stock calculation
            - CSV export of results
            """)

if __name__ == "__main__":
    main()
