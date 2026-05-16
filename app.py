"""
Pragathi Dynamic Predictive & Inventory Optimization Engine
Complete with Google Sheets, Multi-Metric Analysis, and Persistent Graphs
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
import plotly.express as px
import plotly.graph_objects as go

# Google Sheets imports
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GOOGLE_SHEETS_AVAILABLE = True
except ImportError:
    GOOGLE_SHEETS_AVAILABLE = False

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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
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

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'data_loaded': False,
        'df': None,
        'brand_col': None,
        'sales_col': None,
        'inventory_col': None,
        'walkins_col': None,
        'date_col': None,
        'predictions': None,
        'analysis_complete': False,
        'show_pie': False,
        'show_bar': False,
        'show_hist': False,
        'data_source': None,
        'sales_predictions': None,
        'inventory_predictions': None,
        'walkins_predictions': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

def auto_detect_columns(df):
    """Automatically detect brand, sales, inventory, walk-ins, and date columns"""
    detected = {
        'brand_col': None,
        'sales_col': None,
        'inventory_col': None,
        'walkins_col': None,
        'date_col': None
    }
    
    # Brand keywords
    brand_keywords = ['brand', 'make', 'manufacturer', 'company', 'vendor', 'supplier', 'label']
    
    # Sales keywords
    sales_keywords = ['sales', 'revenue', 'turnover', 'order_value', 'sell', 'sold', 'amount']
    
    # Inventory keywords
    inventory_keywords = ['inventory', 'stock', 'on_hand', 'quantity', 'qty', 'available', 'balance']
    
    # Walk-ins keywords
    walkins_keywords = ['walkin', 'walk_in', 'footfall', 'visitor', 'customer_count', 'traffic', 'foot_traffic', 'store_visits']
    
    # Date keywords
    date_keywords = ['date', 'time', 'day', 'month', 'year', 'period', 'timestamp']
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Detect brand column
        if not detected['brand_col']:
            for keyword in brand_keywords:
                if keyword in col_lower:
                    detected['brand_col'] = col
                    break
        
        # Detect sales column
        if not detected['sales_col']:
            for keyword in sales_keywords:
                if keyword in col_lower and df[col].dtype in ['int64', 'float64']:
                    detected['sales_col'] = col
                    break
        
        # Detect inventory column
        if not detected['inventory_col']:
            for keyword in inventory_keywords:
                if keyword in col_lower and df[col].dtype in ['int64', 'float64']:
                    detected['inventory_col'] = col
                    break
        
        # Detect walk-ins column
        if not detected['walkins_col']:
            for keyword in walkins_keywords:
                if keyword in col_lower and df[col].dtype in ['int64', 'float64']:
                    detected['walkins_col'] = col
                    break
        
        # Detect date column
        if not detected['date_col']:
            for keyword in date_keywords:
                if keyword in col_lower:
                    detected['date_col'] = col
                    break
    
    # Fallback: if no sales column found, use first numeric column
    if not detected['sales_col']:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            detected['sales_col'] = numeric_cols[0]
    
    return detected

def clean_data(df):
    """Clean and prepare data"""
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

def load_from_google_sheets(sheet_url):
    """Load data from Google Sheets"""
    if not GOOGLE_SHEETS_AVAILABLE:
        st.error("Google Sheets support not available. Please install: pip install gspread google-auth")
        return None
    
    try:
        # Extract sheet ID from URL
        import re
        pattern = r'/d/([a-zA-Z0-9-_]+)'
        match = re.search(pattern, sheet_url)
        if not match:
            st.error("Invalid Google Sheets URL")
            return None
        
        sheet_id = match.group(1)
        
        # Try to read as public sheet (no authentication required)
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        df = pd.read_csv(csv_url)
        return df
    except Exception as e:
        st.error(f"Error loading Google Sheet: {str(e)}")
        st.info("Make sure the sheet is shared as 'Anyone with the link can view'")
        return None

def train_prediction_model(df, target_col, brand_col=None):
    """Train model to predict a specific target"""
    if target_col not in df.columns:
        return None, None, None, None
    
    # Prepare data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Encode categorical columns
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    
    X = X.fillna(0)
    
    # Split and train
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict
        predictions = model.predict(X)
        mae = mean_absolute_error(y_test, model.predict(X_test))
        r2 = r2_score(y_test, model.predict(X_test))
        
        return model, predictions, mae, r2
    except Exception as e:
        st.warning(f"Could not train model for {target_col}: {str(e)}")
        return None, None, None, None

def analyze_trend(df, col, date_col=None):
    """Analyze trend and provide reasons"""
    if col not in df.columns:
        return None
    
    values = df[col].dropna()
    if len(values) < 2:
        return {"trend": "Insufficient data"}
    
    first_val = values.iloc[0]
    last_val = values.iloc[-1]
    
    if last_val > first_val:
        percent_change = ((last_val - first_val) / first_val) * 100
        return {
            "trend": f"📈 Increasing",
            "change": f"+{percent_change:.1f}%",
            "reason": f"Your {col} is growing. This could be due to increased demand, successful marketing, or seasonal factors."
        }
    elif last_val < first_val:
        percent_change = ((first_val - last_val) / first_val) * 100
        return {
            "trend": f"📉 Decreasing",
            "change": f"-{percent_change:.1f}%",
            "reason": f"Your {col} is declining. Consider checking inventory levels, pricing, or market conditions."
        }
    else:
        return {
            "trend": f"➡️ Stable",
            "change": "0%",
            "reason": f"Your {col} remains stable, indicating consistent patterns."
        }

def analyze_by_brand(df, brand_col, sales_col, inventory_col=None, walkins_col=None):
    """Analyze performance by brand"""
    if not brand_col or brand_col not in df.columns:
        return None
    
    brand_data = df.groupby(brand_col).agg({
        sales_col: ['sum', 'mean', 'count'] if sales_col in df.columns else []
    }).round(2)
    
    brand_data.columns = ['Total_Sales', 'Avg_Sales', 'Record_Count']
    brand_data = brand_data.sort_values('Total_Sales', ascending=False)
    
    return brand_data

def main():
    """Main application"""
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 Pragathi Dynamic Predictive & Inventory Optimization Engine</h1>
        <p>AI-Powered Multi-Metric Analysis | Sales | Inventory | Walk-ins</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data source selection
    st.markdown("### 📂 Data Source")
    
    data_source = st.radio(
        "Choose data source:",
        ["📁 Upload File (CSV/Excel)", "🌐 Google Sheets URL"],
        horizontal=True
    )
    
    df = None
    
    # File upload
    if data_source == "📁 Upload File (CSV/Excel)":
        uploaded_file = st.file_uploader(
            "Drag and drop your CSV or Excel file",
            type=['csv', 'xlsx'],
            help="Upload your sales, inventory, and walk-ins data"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.success(f"✅ File loaded: {len(df)} rows, {len(df.columns)} columns")
                st.session_state.data_source = "file"
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
    
    # Google Sheets
    else:
        sheet_url = st.text_input(
            "Enter Google Sheets URL",
            placeholder="https://docs.google.com/spreadsheets/d/your-sheet-id/edit",
            help="Make sure the sheet is shared as 'Anyone with the link can view'"
        )
        
        if sheet_url:
            with st.spinner("Loading data from Google Sheets..."):
                df = load_from_google_sheets(sheet_url)
                if df is not None:
                    st.success(f"✅ Google Sheet loaded: {len(df)} rows, {len(df.columns)} columns")
                    st.session_state.data_source = "google_sheets"
    
    # If data is loaded, process it
    if df is not None:
        # Clean data
        with st.spinner("🧹 Cleaning and preparing data..."):
            df = clean_data(df)
        
        # Show preview
        with st.expander("👁️ Data Preview (First 5 rows)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)
        
        # Auto-detect columns
        detected = auto_detect_columns(df)
        
        # Display detected columns
        st.markdown("### 🔍 Automatically Detected Columns")
        
        col1, col2 = st.columns(2)
        with col1:
            if detected['brand_col']:
                st.success(f"🏷️ **Brand Column:** {detected['brand_col']}")
            else:
                st.info("ℹ️ No brand column detected")
            
            if detected['sales_col']:
                st.success(f"💰 **Sales Column:** {detected['sales_col']}")
            else:
                st.warning("⚠️ No sales column detected")
        
        with col2:
            if detected['inventory_col']:
                st.success(f"📦 **Inventory Column:** {detected['inventory_col']}")
            else:
                st.info("ℹ️ No inventory column detected")
            
            if detected['walkins_col']:
                st.success(f"🚶 **Walk-ins Column:** {detected['walkins_col']}")
            else:
                st.info("ℹ️ No walk-ins column detected")
        
        # Run analysis button
        if st.button("🚀 Run Complete Analysis", use_container_width=True):
            with st.spinner("Analyzing Sales, Inventory, and Walk-ins..."):
                
                # Store detected columns in session state
                st.session_state.brand_col = detected['brand_col']
                st.session_state.sales_col = detected['sales_col']
                st.session_state.inventory_col = detected['inventory_col']
                st.session_state.walkins_col = detected['walkins_col']
                st.session_state.date_col = detected['date_col']
                st.session_state.df = df
                
                # Train models for each metric
                results = {}
                
                # Sales prediction
                if detected['sales_col']:
                    model, predictions, mae, r2 = train_prediction_model(df, detected['sales_col'], detected['brand_col'])
                    if predictions is not None:
                        results['sales'] = {
                            'model': model,
                            'predictions': predictions,
                            'mae': mae,
                            'r2': r2,
                            'col': detected['sales_col']
                        }
                        df[f'Predicted_{detected["sales_col"]}'] = predictions
                
                # Inventory prediction
                if detected['inventory_col']:
                    model, predictions, mae, r2 = train_prediction_model(df, detected['inventory_col'], detected['brand_col'])
                    if predictions is not None:
                        results['inventory'] = {
                            'model': model,
                            'predictions': predictions,
                            'mae': mae,
                            'r2': r2,
                            'col': detected['inventory_col']
                        }
                        df[f'Predicted_{detected["inventory_col"]}'] = predictions
                
                # Walk-ins prediction
                if detected['walkins_col']:
                    model, predictions, mae, r2 = train_prediction_model(df, detected['walkins_col'], detected['brand_col'])
                    if predictions is not None:
                        results['walkins'] = {
                            'model': model,
                            'predictions': predictions,
                            'mae': mae,
                            'r2': r2,
                            'col': detected['walkins_col']
                        }
                        df[f'Predicted_{detected["walkins_col"]}'] = predictions
                
                st.session_state.predictions = df
                st.session_state.analysis_complete = True
                st.session_state.results = results
        
        # Show results if analysis is complete
        if st.session_state.analysis_complete:
            df = st.session_state.predictions
            results = st.session_state.results
            detected = {
                'brand_col': st.session_state.brand_col,
                'sales_col': st.session_state.sales_col,
                'inventory_col': st.session_state.inventory_col,
                'walkins_col': st.session_state.walkins_col,
                'date_col': st.session_state.date_col
            }
            
            st.success("✅ Analysis Complete!")
            
            # ============ BRAND ANALYSIS SECTION ============
            if detected['brand_col'] and detected['sales_col']:
                st.markdown("### 🏷️ Brand Performance Analysis")
                
                brand_analysis = analyze_by_brand(df, detected['brand_col'], detected['sales_col'])
                if brand_analysis is not None:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Top Brand", brand_analysis.index[0])
                        st.metric("Top Brand Sales", f"{brand_analysis.iloc[0]['Total_Sales']:,.0f}")
                    with col2:
                        st.metric("Total Brands", len(brand_analysis))
                        st.metric("Average Sales/Brand", f"{brand_analysis['Total_Sales'].mean():,.0f}")
                    
                    st.dataframe(brand_analysis.head(10), use_container_width=True)
            
            # ============ THREE METRICS SUMMARY ============
            st.markdown("### 📊 Multi-Metric Analysis Summary")
            
            metric_cols = st.columns(3)
            
            # Sales Metric
            with metric_cols[0]:
                if 'sales' in results:
                    sales_trend = analyze_trend(df, detected['sales_col'], detected['date_col'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>💰 SALES</h3>
                        <h2>{sales_trend['trend']}</h2>
                        <p>Change: {sales_trend['change']}</p>
                        <p>MAE: {results['sales']['mae']:.2f}</p>
                        <p>R²: {results['sales']['r2']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No sales data available")
            
            # Inventory Metric
            with metric_cols[1]:
                if 'inventory' in results:
                    inv_trend = analyze_trend(df, detected['inventory_col'], detected['date_col'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>📦 INVENTORY</h3>
                        <h2>{inv_trend['trend']}</h2>
                        <p>Change: {inv_trend['change']}</p>
                        <p>MAE: {results['inventory']['mae']:.2f}</p>
                        <p>R²: {results['inventory']['r2']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No inventory data available")
            
            # Walk-ins Metric
            with metric_cols[2]:
                if 'walkins' in results:
                    walk_trend = analyze_trend(df, detected['walkins_col'], detected['date_col'])
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>🚶 WALK-INS</h3>
                        <h2>{walk_trend['trend']}</h2>
                        <p>Change: {walk_trend['change']}</p>
                        <p>MAE: {results['walkins']['mae']:.2f}</p>
                        <p>R²: {results['walkins']['r2']:.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("No walk-ins data available")
            
            # ============ REASON ANALYSIS ============
            st.markdown("### 🔍 Trend Analysis & Reasons")
            
            if 'sales' in results and detected['sales_col']:
                sales_trend = analyze_trend(df, detected['sales_col'], detected['date_col'])
                st.markdown(f"""
                <div class="info-card">
                    <strong>💰 Sales Trend:</strong> {sales_trend['trend']} ({sales_trend['change']})<br>
                    💡 {sales_trend['reason']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'inventory' in results and detected['inventory_col']:
                inv_trend = analyze_trend(df, detected['inventory_col'], detected['date_col'])
                st.markdown(f"""
                <div class="info-card">
                    <strong>📦 Inventory Trend:</strong> {inv_trend['trend']} ({inv_trend['change']})<br>
                    💡 {inv_trend['reason']}
                </div>
                """, unsafe_allow_html=True)
            
            if 'walkins' in results and detected['walkins_col']:
                walk_trend = analyze_trend(df, detected['walkins_col'], detected['date_col'])
                st.markdown(f"""
                <div class="info-card">
                    <strong>🚶 Walk-ins Trend:</strong> {walk_trend['trend']} ({walk_trend['change']})<br>
                    💡 {walk_trend['reason']}
                </div>
                """, unsafe_allow_html=True)
            
            # ============ PREDICTION TABLES ============
            st.markdown("### 📈 Predictions")
            
            # Create tabs for different metrics
            pred_tabs = st.tabs(["💰 Sales Predictions", "📦 Inventory Predictions", "🚶 Walk-ins Predictions"])
            
            with pred_tabs[0]:
                if 'sales' in results and detected['sales_col']:
                    display_cols = [detected['sales_col'], f'Predicted_{detected["sales_col"]}']
                    if detected['brand_col']:
                        display_cols = [detected['brand_col']] + display_cols
                    st.dataframe(df[display_cols].head(15), use_container_width=True)
                else:
                    st.info("No sales predictions available")
            
            with pred_tabs[1]:
                if 'inventory' in results and detected['inventory_col']:
                    display_cols = [detected['inventory_col'], f'Predicted_{detected["inventory_col"]}']
                    if detected['brand_col']:
                        display_cols = [detected['brand_col']] + display_cols
                    st.dataframe(df[display_cols].head(15), use_container_width=True)
                else:
                    st.info("No inventory predictions available")
            
            with pred_tabs[2]:
                if 'walkins' in results and detected['walkins_col']:
                    display_cols = [detected['walkins_col'], f'Predicted_{detected["walkins_col"]}']
                    if detected['brand_col']:
                        display_cols = [detected['brand_col']] + display_cols
                    st.dataframe(df[display_cols].head(15), use_container_width=True)
                else:
                    st.info("No walk-ins predictions available")
            
            # ============ OPTIONAL CHARTS (Checkboxes - Now Persistent!) ============
            st.markdown("### 📊 Optional Charts")
            st.info("💡 Check the boxes below to view charts")
            
            # These checkboxes now use session_state to persist
            show_pie = st.checkbox("🥧 Show Procurement Status Pie Chart", 
                                   value=st.session_state.show_pie,
                                   key="pie_chart")
            st.session_state.show_pie = show_pie
            
            show_bar = st.checkbox("📊 Show Top Items to Order (Bar Graph)",
                                  value=st.session_state.show_bar,
                                  key="bar_chart")
            st.session_state.show_bar = show_bar
            
            show_hist = st.checkbox("📉 Show Sales Distribution (Histogram)",
                                   value=st.session_state.show_hist,
                                   key="hist_chart")
            st.session_state.show_hist = show_hist
            
            # Pie Chart
            if show_pie and detected['inventory_col']:
                # Create inventory status
                inv_col = detected['inventory_col']
                threshold = df[inv_col].mean()
                df['Inventory_Status'] = df[inv_col].apply(
                    lambda x: "⚠️ Low Stock" if x < threshold * 0.3 
                    else "🟡 Medium Stock" if x < threshold * 0.7 
                    else "✅ High Stock"
                )
                
                status_counts = df['Inventory_Status'].value_counts()
                fig_pie = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title='Inventory Status Distribution',
                    color_discrete_map={
                        '✅ High Stock': '#4CAF50',
                        '🟡 Medium Stock': '#FFC107',
                        '⚠️ Low Stock': '#FF6B6B'
                    },
                    hole=0.3
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=450)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Bar Chart - Top items
            if show_bar and detected['sales_col']:
                if detected['brand_col']:
                    top_brands = df.groupby(detected['brand_col'])[detected['sales_col']].sum().nlargest(10).reset_index()
                    fig_bar = px.bar(
                        top_brands,
                        x=detected['brand_col'],
                        y=detected['sales_col'],
                        title='Top 10 Brands by Sales',
                        labels={detected['brand_col']: 'Brand', detected['sales_col']: 'Total Sales'},
                        color=detected['sales_col'],
                        color_continuous_scale='Viridis',
                        text=detected['sales_col']
                    )
                    fig_bar.update_traces(texttemplate='%{text:.0f}', textposition='outside')
                    fig_bar.update_layout(height=450)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.info("No brand column detected for bar chart")
            
            # Histogram
            if show_hist and detected['sales_col']:
                fig_hist = px.histogram(
                    df,
                    x=detected['sales_col'],
                    nbins=20,
                    title=f'Distribution of {detected["sales_col"]}',
                    color_discrete_sequence=['#667eea'],
                    labels={detected['sales_col']: 'Sales Value', 'count': 'Frequency'}
                )
                fig_hist.update_layout(height=450)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # ============ INVENTORY OPTIMIZATION (if inventory column exists) ============
            if detected['inventory_col'] and detected['sales_col']:
                st.markdown("### 📦 Inventory Optimization Recommendations")
                
                # Calculate safety stock and reorder point
                avg_demand = df[detected['sales_col']].mean()
                demand_std = df[detected['sales_col']].std()
                z_score = 1.65  # 95% service level
                
                df['Safety_Stock'] = z_score * demand_std
                df['Reorder_Point'] = avg_demand + df['Safety_Stock']
                
                # Determine which items need ordering
                if detected['inventory_col'] in df.columns:
                    df['Needs_Reorder'] = df[detected['inventory_col']] < df['Reorder_Point']
                    reorder_count = df['Needs_Reorder'].sum()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Items Needing Reorder", reorder_count)
                    with col2:
                        st.metric("Recommended Safety Stock", f"{df['Safety_Stock'].mean():.0f}")
                    
                    # Show reorder recommendations
                    reorder_df = df[df['Needs_Reorder']].head(10)
                    if len(reorder_df) > 0:
                        display_cols = []
                        if detected['brand_col']:
                            display_cols.append(detected['brand_col'])
                        display_cols.extend([detected['inventory_col'], 'Reorder_Point', 'Safety_Stock'])
                        st.warning(f"⚠️ {reorder_count} items need immediate reordering!")
                        st.dataframe(reorder_df[display_cols], use_container_width=True)
            
            # ============ CORRELATION ANALYSIS ============
            if detected['sales_col'] and detected['walkins_col']:
                st.markdown("### 🔗 Correlation Analysis: Sales vs Walk-ins")
                
                # Calculate correlation
                correlation = df[detected['sales_col']].corr(df[detected['walkins_col']])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Correlation Coefficient", f"{correlation:.3f}")
                    if correlation > 0.7:
                        st.success("✅ Strong positive correlation - More walk-ins lead to more sales")
                    elif correlation > 0.3:
                        st.info("📊 Moderate correlation - Walk-ins influence sales")
                    elif correlation > 0:
                        st.warning("⚠️ Weak correlation - Other factors affect sales")
                    else:
                        st.error("❌ Negative correlation - Walk-ins don't drive sales")
                
                with col2:
                    # Scatter plot
                    fig_scatter = px.scatter(
                        df,
                        x=detected['walkins_col'],
                        y=detected['sales_col'],
                        title='Sales vs Walk-ins Relationship',
                        labels={detected['walkins_col']: 'Walk-ins', detected['sales_col']: 'Sales'},
                        trendline='ols',
                        color_discrete_sequence=['#667eea']
                    )
                    fig_scatter.update_layout(height=300)
                    st.plotly_chart(fig_scatter, use_container_width=True)
            
            # ============ DOWNLOAD RESULTS ============
            st.markdown("### 📥 Export Results")
            
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue().encode()
            
            st.download_button(
                "📥 Download Complete Results (CSV)",
                csv_data,
                f"pragathi_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
            
            st.info("✅ Analysis complete! Checkboxes above let you view additional charts.")
    
    else:
        # Welcome message when no data loaded
        st.info("👈 **Get Started:** Upload a file or enter a Google Sheets URL above")
        
        with st.expander("📋 Sample Data Format", expanded=True):
            st.markdown("""
            ### Your data should have columns like:
            
            | Brand | Date | Sales | Inventory | Walk-ins |
            |-------|------|-------|-----------|----------|
            | Nike | 2024-01-01 | 5000 | 200 | 150 |
            | Adidas | 2024-01-01 | 4500 | 180 | 130 |
            | Puma | 2024-01-02 | 3000 | 120 | 90 |
            
            ### The app will automatically detect:
            - 🏷️ **Brand column** (brand, make, manufacturer)
            - 💰 **Sales column** (sales, revenue, amount)
            - 📦 **Inventory column** (stock, inventory, quantity)
            - 🚶 **Walk-ins column** (walkins, footfall, visitors)
            - 📅 **Date column** (date, time, day)
            
            ### Features:
            - ✅ Automatic detection - no manual selection needed
            - ✅ Analyzes ALL metrics simultaneously
            - ✅ Predicts future values for Sales, Inventory, Walk-ins
            - ✅ Shows trend analysis with reasons
            - ✅ Brand performance breakdown
            - ✅ Correlation between walk-ins and sales
            - ✅ Persistent charts (don't disappear on checkbox click)
            """)

if __name__ == "__main__":
    main()
