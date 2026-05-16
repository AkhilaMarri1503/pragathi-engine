"""
Pragathi Dynamic Predictive & Inventory Optimization Engine
Production-ready Supply Chain Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import re
import hashlib
import sqlite3
import json
import logging
import traceback
from typing import Tuple, Optional, Dict, Any, List
from pathlib import Path
import time
import warnings

# Suppress warnings for cleaner UI
warnings.filterwarnings('ignore')

# Data processing
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, KFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Time Series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

# SHAP for explainability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# PDF Export
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# Password hashing
try:
    from passlib.hash import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    # Fallback to simple hashing
    import hashlib
    def bcrypt_hash(password):
        return hashlib.sha256(password.encode()).hexdigest()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Page configuration
st.set_page_config(
    page_title="Pragathi Analysis Engine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 0.5rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
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
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .highlight {
        background-color: #ffcccc;
        padding: 0.25rem;
        border-radius: 0.25rem;
    }
    .fade-in {
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'authenticated': False,
        'username': None,
        'user_role': 'viewer',
        'login_time': None,
        'last_activity': None,
        'data': None,
        'cleaned_data': None,
        'target_column': None,
        'detected_target': None,
        'target_confidence': None,
        'analysis_type': None,
        'model': None,
        'predictions': None,
        'inventory_results': None,
        'forecast_results': None,
        'cleaning_summary': None,
        'theme': 'light',
        'language': 'English',
        'sidebar_page': 'Home',
        'inventory_z_score': 1.65,
        'google_calendar_enabled': False,
        'google_calendar_data': None,
        'holiday_features': None,
        'logs': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Database setup
def init_database():
    """Initialize SQLite database for user management"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        )
    ''')
    
    # Check if admin exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        
        if BCRYPT_AVAILABLE:
            password_hash = bcrypt.hash("admin123")
        else:
            password_hash = bcrypt_hash("admin123")
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ('admin', password_hash, 'admin', timestamp)
        )
        conn.commit()
    
    conn.close()

init_database()

def verify_password(username: str, password: str) -> bool:
    """Verify user password"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    
    if result:
        stored_hash = result[0]
        if BCRYPT_AVAILABLE:
            return bcrypt.verify(password, stored_hash)
        else:
            return stored_hash == bcrypt_hash(password)
    return False

def get_user_role(username: str) -> str:
    """Get user role from database"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else 'viewer'

def update_last_login(username: str):
    """Update user's last login timestamp"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET last_login = ? WHERE username = ?",
        (datetime.now().isoformat(), username)
    )
    conn.commit()
    conn.close()

def add_user(username: str, password: str, role: str) -> bool:
    """Add a new user to the database"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        # Check if user exists
        cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            return False
        
        timestamp = datetime.now().isoformat()
        
        if BCRYPT_AVAILABLE:
            password_hash = bcrypt.hash(password)
        else:
            password_hash = bcrypt_hash(password)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, role, timestamp)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error adding user: {e}")
        return False

def delete_user(username: str) -> bool:
    """Delete a user from the database"""
    if username == 'admin':
        return False
    
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error deleting user: {e}")
        return False

def change_password(username: str, new_password: str) -> bool:
    """Change user password"""
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        
        if BCRYPT_AVAILABLE:
            password_hash = bcrypt.hash(new_password)
        else:
            password_hash = bcrypt_hash(new_password)
        
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logging.error(f"Error changing password: {e}")
        return False

def login_ui():
    """Display login screen"""
    st.markdown("""
    <div class="main-header" style="padding: 3rem; margin-bottom: 2rem;">
        <h1 style="font-size: 3rem;">🚀 Pragathi Analysis Engine</h1>
        <p style="font-size: 1.2rem;">AI-Powered Predictive & Inventory Optimization Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        remember = st.checkbox("Remember me")
        
        if st.button("🚀 Login", use_container_width=True):
            if verify_password(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = get_user_role(username)
                st.session_state.login_time = datetime.now()
                update_last_login(username)
                st.rerun()
            else:
                st.error("❌ Invalid username or password")
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center;">
            <p style="color: #666;">Default admin credentials:</p>
            <code>Username: admin</code><br>
            <code>Password: admin123</code>
        </div>
        """, unsafe_allow_html=True)

def sidebar_navigation():
    """Display sidebar navigation"""
    with st.sidebar:
        # Theme toggle
        theme = st.selectbox(
            "🎨 Theme",
            ["Light", "Dark"],
            index=0 if st.session_state.theme == 'light' else 1
        )
        st.session_state.theme = theme.lower()
        
        # Language selection
        language = st.selectbox(
            "🌐 Language",
            ["English", "Spanish", "French", "Hindi"],
            index=0
        )
        st.session_state.language = language
        
        st.markdown("---")
        
        # Navigation
        pages = ["🏠 Home", "📈 Analysis", "📊 Inventory", "📄 Export"]
        
        if st.session_state.user_role == 'admin':
            pages.append("⚙️ Settings")
        
        selected = st.radio("Navigation", pages)
        st.session_state.sidebar_page = selected
        
        st.markdown("---")
        
        # Google Calendar enrichment
        st.session_state.google_calendar_enabled = st.checkbox(
            "🎯 Enrich with Google Calendar",
            value=st.session_state.google_calendar_enabled,
            help="Add holiday features to improve predictions"
        )
        
        # Z-score slider for inventory
        st.session_state.inventory_z_score = st.slider(
            "🔒 Service Level (Z-score)",
            min_value=0.50,
            max_value=3.00,
            step=0.05,
            value=st.session_state.inventory_z_score,
            help="Higher Z-score = more safety stock but higher carrying cost"
        )
        
        service_level = {
            1.65: "95%",
            1.96: "97.5%",
            2.33: "99%",
            2.58: "99.5%"
        }.get(st.session_state.inventory_z_score, f"{100 - (50 / st.session_state.inventory_z_score):.1f}%")
        
        st.caption(f"Currently: {service_level} service level protection")
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.caption(f"Logged in as: **{st.session_state.username}**")
        st.caption(f"Role: **{st.session_state.user_role}**")

def auto_detect_target(df: pd.DataFrame, date_col: str = None, inv_cols: Dict = None) -> Tuple[Optional[str], str, List]:
    """
    Automatically detect the best target column for prediction
    
    Returns: (target_column, confidence_level, candidates_with_scores)
    """
    # Define exclusion patterns
    exclude_patterns = [
        r'.*id$', r'.*_id$', r'^id$', r'index', r'key', r'row',
        r'.*_date$', r'.*time.*', r'timestamp',
        r'.*stock.*', r'.*inventory.*', r'.*lead.*', r'.*sku.*',
        r'.*code$', r'.*name$', r'.*description$'
    ]
    
    # Define target keywords with scores
    target_keywords = {
        'exact': {
            'target': 10, 'output': 10, 'label': 10,
            'sales': 8, 'demand': 8, 'revenue': 8, 'quantity': 8,
            'qty': 6, 'price': 6, 'amount': 6, 'value': 6, 'cost': 6,
            'volume': 4, 'units': 4, 'count': 4, 'total': 4,
            'net': 2, 'gross': 2, 'sum': 2, 'aggregate': 2
        }
    }
    
    # Exclude detected date column
    exclude_cols = []
    if date_col:
        exclude_cols.append(date_col)
    
    # Exclude inventory columns
    if inv_cols:
        for key, col in inv_cols.items():
            if col:
                exclude_cols.append(col)
    
    # Exclude columns by patterns
    for col in df.columns:
        col_lower = col.lower()
        for pattern in exclude_patterns:
            if re.match(pattern, col_lower):
                exclude_cols.append(col)
                break
    
    # Also exclude columns with >50% unique values (likely IDs)
    for col in df.columns:
        if col not in exclude_cols and df[col].nunique() / len(df) > 0.5:
            exclude_cols.append(col)
    
    # Get candidate columns (numeric, not excluded)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    candidates = [col for col in numeric_cols if col not in exclude_cols]
    
    # Score candidates
    scores = {}
    for col in candidates:
        score = 0
        col_lower = col.lower()
        
        # Check for exact matches
        for keyword, points in target_keywords['exact'].items():
            if col_lower == keyword:
                score += points
            elif keyword in col_lower:
                score += points // 2
        
        # Penalize low variance
        variance = df[col].var()
        if variance < 0.01:
            score -= 2
        
        scores[col] = score
    
    # Sort candidates by score
    sorted_candidates = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Select best candidate
    if sorted_candidates:
        best_col, best_score = sorted_candidates[0]
        
        # Determine confidence
        if best_score >= 8:
            confidence = "High 🟢"
        elif best_score >= 5:
            confidence = "Medium 🟡"
        elif best_score >= 2:
            confidence = "Low 🟠"
        else:
            confidence = "None 🔴"
        
        return best_col, confidence, sorted_candidates
    else:
        return None, "None 🔴", []

def auto_detect_inventory_columns(df: pd.DataFrame) -> Dict:
    """Auto-detect inventory-related columns"""
    patterns = {
        'stock_col': [
            'current_stock', 'stock', 'inventory', 'on_hand', 
            'qty_on_hand', 'available', 'quantity_on_hand', 'stock_level'
        ],
        'lead_time_col': [
            'lead_time', 'lead time', 'lead', 'delivery_days',
            'replenishment_days', 'supplier_lead'
        ],
        'sku_col': [
            'sku', 'item', 'product', 'material', 'part',
            'code', 'product_id', 'item_code', 'article'
        ]
    }
    
    detected = {}
    for key, pattern_list in patterns.items():
        found = None
        for col in df.columns:
            col_lower = col.lower()
            for pattern in pattern_list:
                if pattern in col_lower:
                    found = col
                    break
            if found:
                break
        detected[key] = found
    
    return detected

def auto_detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the most likely date column"""
    date_patterns = ['date', 'time', 'timestamp', 'transaction', 'created', 
                    'modified', 'datetime', 'period', 'day', 'month', 'year']
    
    best_col = None
    best_score = 0
    
    for col in df.columns:
        col_lower = col.lower()
        score = 0
        
        for pattern in date_patterns:
            if pattern in col_lower:
                score += 1
        
        # Try to convert to datetime
        try:
            pd.to_datetime(df[col], errors='coerce')
            if df[col].dtype == 'object':
                score += 2
        except:
            pass
        
        if score > best_score:
            best_score = score
            best_col = col
    
    return best_col if best_score > 0 else None

def detect_analysis_type(df: pd.DataFrame, target_col: str) -> str:
    """Auto-detect analysis type: Regression, Classification, or Time Series"""
    
    target_data = df[target_col]
    
    # Check if classification
    if target_data.dtype == 'object':
        return 'Classification'
    
    if target_data.dtype in ['int64', 'int32'] and target_data.nunique() < 15:
        return 'Classification'
    
    if target_data.nunique() < 0.05 * len(df):
        return 'Classification'
    
    return 'Regression'

def clean_data(df: pd.DataFrame, progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """Automated data cleaning engine"""
    summary = {
        'initial_rows': len(df),
        'initial_cols': len(df.columns),
        'empty_rows_removed': 0,
        'empty_cols_removed': 0,
        'duplicates_removed': 0,
        'missing_values_filled': 0,
        'outliers_detected': 0,
        'date_col_detected': None,
        'inventory_cols': {}
    }
    
    # Step 1: Remove empty rows and columns
    if progress_callback:
        progress_callback(0.1, "Removing empty rows and columns...")
    
    before_rows = len(df)
    df = df.dropna(how='all')
    summary['empty_rows_removed'] = before_rows - len(df)
    
    before_cols = len(df.columns)
    df = df.dropna(axis=1, how='all')
    summary['empty_cols_removed'] = before_cols - len(df.columns)
    
    # Step 2: Remove duplicates
    before_rows = len(df)
    df = df.drop_duplicates()
    summary['duplicates_removed'] = before_rows - len(df)
    
    # Step 3: Detect date column
    if progress_callback:
        progress_callback(0.2, "Detecting date columns...")
    
    date_col = auto_detect_date_column(df)
    if date_col:
        summary['date_col_detected'] = date_col
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Extract temporal features
        df[f'{date_col}_Month'] = df[date_col].dt.month
        df[f'{date_col}_Week_of_Year'] = df[date_col].dt.isocalendar().week
        df[f'{date_col}_Day_of_Week'] = df[date_col].dt.dayofweek
        df[f'{date_col}_Quarter'] = df[date_col].dt.quarter
        df[f'{date_col}_Is_Weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        df[f'{date_col}_Is_Month_Start'] = df[date_col].dt.is_month_start.astype(int)
        df[f'{date_col}_Is_Month_End'] = df[date_col].dt.is_month_end.astype(int)
    
    # Step 4: Detect inventory columns
    if progress_callback:
        progress_callback(0.3, "Detecting inventory columns...")
    
    summary['inventory_cols'] = auto_detect_inventory_columns(df)
    
    # Step 5: Fill missing values
    if progress_callback:
        progress_callback(0.4, "Filling missing values...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    filled_count = 0
    for col in numeric_cols:
        if df[col].isnull().any():
            # Try interpolation first
            if len(df) > 1:
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            # Fill remaining with median
            remaining_nulls = df[col].isnull().sum()
            if remaining_nulls > 0:
                df[col].fillna(df[col].median(), inplace=True)
                filled_count += remaining_nulls
    
    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                null_count = df[col].isnull().sum()
                df[col].fillna(mode_val[0], inplace=True)
                filled_count += null_count
    
    summary['missing_values_filled'] = filled_count
    
    # Step 6: Detect outliers using IQR
    if progress_callback:
        progress_callback(0.5, "Detecting outliers...")
    
    outlier_flags = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        flags = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
        outlier_flags.append(flags)
        summary['outliers_detected'] += flags.sum()
    
    if outlier_flags:
        df['Outlier_Flag'] = sum(outlier_flags) > 0
    
    summary['final_rows'] = len(df)
    summary['final_cols'] = len(df.columns)
    
    return df, summary

def load_and_clean_file(uploaded_file, progress_callback=None) -> Tuple[pd.DataFrame, Dict]:
    """Load and clean uploaded file with chunking for large files"""
    
    file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
    
    if file_size > 100:
        # Use chunking for large files
        st.info(f"📦 Large file detected ({file_size:.1f} MB). Processing in chunks...")
        
        chunks = []
        if uploaded_file.name.endswith('.csv'):
            for chunk in pd.read_csv(uploaded_file, chunksize=50000):
                chunks.append(chunk)
        else:
            # Excel files loaded at once
            df = pd.read_excel(uploaded_file)
            chunks = [df]
        
        df = pd.concat(chunks, ignore_index=True)
    else:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    
    # Clean the data
    cleaned_df, summary = clean_data(df, progress_callback)
    
    return cleaned_df, summary

def train_models(df: pd.DataFrame, target_col: str, analysis_type: str, date_col: str = None, progress_callback=None):
    """Train machine learning models based on analysis type"""
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Handle categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(0)
    
    # Split data
    if analysis_type == 'Time Series' and date_col:
        # Use TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        # Use first 80% for training
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    models = {}
    results = []
    
    if analysis_type == 'Regression':
        if progress_callback:
            progress_callback(0.3, "Training regression models...")
        
        regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(),
            'Lasso': Lasso(),
            'Elastic Net': ElasticNet(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
        }
        
        for name, model in regression_models.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                models[name] = model
                results.append({
                    'Model': name,
                    'MAE': mae,
                    'RMSE': rmse,
                    'R2': r2,
                    'Training Time (s)': round(train_time, 2)
                })
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
        
        # Sort by MAE
        results_df = pd.DataFrame(results).sort_values('MAE')
        best_model_name = results_df.iloc[0]['Model']
        best_model = models[best_model_name]
        
        # Predict on full dataset
        full_predictions = best_model.predict(X)
        
    elif analysis_type == 'Classification':
        if progress_callback:
            progress_callback(0.3, "Training classification models...")
        
        classification_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBClassifier(random_state=42, n_jobs=-1),
            'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1),
            'Naive Bayes': GaussianNB()
        }
        
        for name, model in classification_models.items():
            try:
                start_time = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start_time
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                
                models[name] = model
                results.append({
                    'Model': name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'Training Time (s)': round(train_time, 2)
                })
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
        
        # Sort by Accuracy
        results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
        best_model_name = results_df.iloc[0]['Model']
        best_model = models[best_model_name]
        
        # Predict on full dataset
        full_predictions = best_model.predict(X)
        if hasattr(best_model, 'predict_proba'):
            full_proba = best_model.predict_proba(X)
        else:
            full_proba = None
    
    else:  # Time Series
        if PROPHET_AVAILABLE and date_col:
            if progress_callback:
                progress_callback(0.3, "Training Prophet time series model...")
            
            # Prepare data for Prophet
            prophet_df = df[[date_col, target_col]].copy()
            prophet_df = prophet_df.rename(columns={date_col: 'ds', target_col: 'y'})
            prophet_df = prophet_df.dropna()
            prophet_df = prophet_df.sort_values('ds')
            
            # Train Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                interval_width=0.95
            )
            model.fit(prophet_df)
            
            # Create future dataframe (default 30 days)
            future_days = st.session_state.get('forecast_days', 30)
            future = model.make_future_dataframe(periods=future_days)
            forecast = model.predict(future)
            
            # Merge with original data
            df = df.merge(
                forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                left_on=date_col,
                right_on='ds',
                how='left'
            )
            
            full_predictions = df['yhat'].fillna(df[target_col])
            results_df = pd.DataFrame([{
                'Model': 'Prophet',
                'MAE': None,  # Would need holdout evaluation
                'Forecast Days': future_days
            }])
            best_model = model
            best_model_name = 'Prophet'
        else:
            st.error("Prophet is not installed. Please install: pip install prophet")
            return df, None, None, None, []
    
    # Add predictions to dataframe
    df[f'Predicted_{target_col}'] = full_predictions
    
    return df, best_model, best_model_name, results_df, models

def calculate_inventory_optimization(df: pd.DataFrame, target_col: str, inv_cols: Dict, 
                                     date_col: str = None, z_score: float = 1.65) -> pd.DataFrame:
    """Calculate inventory optimization metrics"""
    
    df = df.copy()
    
    # Check if we have required columns
    stock_col = inv_cols.get('stock_col')
    lead_col = inv_cols.get('lead_time_col')
    sku_col = inv_cols.get('sku_col')
    
    if not stock_col:
        return df
    
    # If no SKU column, treat entire dataset as one SKU
    if not sku_col:
        df['_temp_sku'] = 'All Products'
        sku_col = '_temp_sku'
    
    # Initialize columns
    df['Safety_Stock'] = 0
    df['Reorder_Point'] = 0
    df['Procurement_Status'] = '✅ STOCK SUFFICIENT'
    df['Recommended_Order_Qty'] = 0
    df['Demand_Volatility'] = 0
    df['Avg_Weekly_Demand'] = 0
    
    # Group by SKU
    for sku in df[sku_col].unique():
        mask = df[sku_col] == sku
        sku_data = df[mask]
        
        # Calculate weekly demand if date column exists
        if date_col and f'{date_col}_Week_of_Year' in df.columns:
            weekly_demand = sku_data.groupby(f'{date_col}_Week_of_Year')[target_col].sum()
            avg_weekly_demand = weekly_demand.mean()
            volatility = weekly_demand.std()
        else:
            avg_weekly_demand = sku_data[target_col].mean()
            volatility = sku_data[target_col].std()
        
        # Get lead time
        if lead_col and lead_col in df.columns:
            avg_lead_time = sku_data[lead_col].mean()
        else:
            avg_lead_time = 7  # Default 7 days
        
        lead_time_weeks = avg_lead_time / 7
        
        # Safety Stock = Z * Volatility * sqrt(Lead Time Weeks)
        safety_stock = z_score * volatility * np.sqrt(max(lead_time_weeks, 0.1))
        
        # Reorder Point = (Avg Weekly Demand * Lead Time Weeks) + Safety Stock
        reorder_point = (avg_weekly_demand * lead_time_weeks) + safety_stock
        
        # Get current stock
        current_stock = sku_data[stock_col].iloc[0] if len(sku_data) > 0 else 0
        
        # Determine action
        if current_stock < reorder_point:
            procurement_status = "⚠️ TRIGGER SUPPLIER PURCHASE ORDER"
            order_qty = max(0, (avg_weekly_demand * 4) + safety_stock - current_stock)
        else:
            procurement_status = "✅ STOCK SUFFICIENT"
            order_qty = 0
        
        # Assign values back
        df.loc[mask, 'Safety_Stock'] = safety_stock
        df.loc[mask, 'Reorder_Point'] = reorder_point
        df.loc[mask, 'Procurement_Status'] = procurement_status
        df.loc[mask, 'Recommended_Order_Qty'] = order_qty
        df.loc[mask, 'Demand_Volatility'] = volatility
        df.loc[mask, 'Avg_Weekly_Demand'] = avg_weekly_demand
    
    # Clean up temporary column
    if '_temp_sku' in df.columns:
        df = df.drop(columns=['_temp_sku'])
    
    return df

def generate_explanations(results: Dict) -> str:
    """Generate simple English explanations"""
    
    explanations = []
    
    if results.get('model_name'):
        explanations.append(f"**🤖 Your best prediction model is: {results['model_name']}**")
        
        if results.get('mae'):
            explanations.append(f"📊 On average, predictions are off by **{results['mae']:.2f}** units.")
        
        if results.get('r2'):
            r2_pct = results['r2'] * 100
            if r2_pct > 80:
                explanations.append(f"🌟 Great news! The model explains **{r2_pct:.1f}%** of the patterns in your data.")
            elif r2_pct > 50:
                explanations.append(f"📈 The model explains **{r2_pct:.1f}%** of the patterns in your data.")
            else:
                explanations.append(f"⚠️ The model explains only **{r2_pct:.1f}%** of the patterns. Try adding more features.")
    
    if results.get('inventory_summary'):
        inv = results['inventory_summary']
        explanations.append(f"\n**📦 Inventory Summary:**")
        explanations.append(f"🚨 **{inv['trigger_count']}** out of {inv['total_skus']} SKUs need immediate reordering.")
        explanations.append(f"💰 Total recommended order quantity: **{int(inv['total_order_qty'])}** units.")
        
        if inv['top_skus']:
            explanations.append(f"🔥 Highest priority: {', '.join(inv['top_skus'][:3])}")
    
    if results.get('trend'):
        if results['trend'] > 0.05:
            explanations.append(f"\n📈 Your {results['target']} is **increasing** by about {results['trend']*100:.1f}%.")
        elif results['trend'] < -0.05:
            explanations.append(f"\n📉 Your {results['target']} is **decreasing** by about {abs(results['trend']*100):.1f}%. Consider adjusting inventory levels.")
        else:
            explanations.append(f"\n➡️ Your {results['target']} is **stable** - good for inventory planning.")
    
    return "\n".join(explanations)

def create_actual_vs_predicted_plot(df: pd.DataFrame, actual_col: str, pred_col: str):
    """Create actual vs predicted scatter plot"""
    
    fig = go.Figure()
    
    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df[actual_col],
        y=df[pred_col],
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color=np.abs(df[actual_col] - df[pred_col]),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Absolute Error")
        ),
        text=[f"Actual: {a:.2f}<br>Predicted: {p:.2f}<br>Error: {abs(a-p):.2f}"
              for a, p in zip(df[actual_col], df[pred_col])],
        hoverinfo='text'
    ))
    
    # Add perfect prediction line
    min_val = min(df[actual_col].min(), df[pred_col].min())
    max_val = max(df[actual_col].max(), df[pred_col].max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    # Calculate metrics
    mae = mean_absolute_error(df[actual_col], df[pred_col])
    r2 = r2_score(df[actual_col], df[pred_col])
    
    fig.update_layout(
        title=f"Actual vs Predicted (MAE: {mae:.2f}, R²: {r2:.3f})",
        xaxis_title=f"Actual {actual_col}",
        yaxis_title=f"Predicted {pred_col}",
        height=500,
        hovermode='closest'
    )
    
    return fig

def create_feature_importance_plot(model, feature_names: List[str], top_n: int = 10):
    """Create feature importance bar chart"""
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_)
    else:
        return None
    
    # Sort features by importance
    indices = np.argsort(importances)[-top_n:]
    
    fig = go.Figure(go.Bar(
        x=[importances[i] for i in indices],
        y=[feature_names[i] for i in indices],
        orientation='h',
        marker_color='#4CAF50',
        text=[f"{importances[i]:.3f}" for i in indices],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500
    )
    
    return fig

def create_forecast_plot(historical_df: pd.DataFrame, forecast_df: pd.DataFrame, 
                         date_col: str, target_col: str):
    """Create time series forecast plot"""
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df[date_col],
        y=historical_df[target_col],
        mode='lines+markers',
        name='Historical',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange', width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_upper'],
        mode='lines',
        name='Upper Bound',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_df['ds'],
        y=forecast_df['yhat_lower'],
        mode='lines',
        name='Confidence Interval (95%)',
        fill='tonexty',
        fillcolor='rgba(255, 165, 0, 0.2)',
        line=dict(width=0)
    ))
    
    fig.update_layout(
        title=f"Forecast for {target_col}",
        xaxis_title="Date",
        yaxis_title=target_col,
        height=500,
        hovermode='x unified'
    )
    
    return fig

def create_inventory_chart(df: pd.DataFrame, sku_col: str, stock_col: str, rop_col: str):
    """Create stock vs reorder point bar chart"""
    
    if sku_col not in df.columns or len(df[sku_col].unique()) > 20:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df[sku_col],
        y=df[stock_col],
        name='Current Stock',
        marker_color='#4CAF50'
    ))
    
    fig.add_trace(go.Bar(
        x=df[sku_col],
        y=df[rop_col],
        name='Reorder Point',
        marker_color='#ff6b6b'
    ))
    
    fig.update_layout(
        title="Current Stock vs Reorder Point by SKU",
        xaxis_title="SKU",
        yaxis_title="Quantity",
        height=500,
        barmode='group'
    )
    
    return fig

def export_to_csv(df: pd.DataFrame) -> bytes:
    """Export dataframe to CSV"""
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode()

def export_to_pdf(df: pd.DataFrame, results: Dict) -> bytes:
    """Generate PDF report"""
    
    if not FPDF_AVAILABLE:
        return None
    
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", size=24, style='B')
    pdf.cell(200, 20, txt="Pragathi Analysis Report", ln=1, align='C')
    
    # Date
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=1, align='C')
    pdf.ln(10)
    
    # Summary
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Executive Summary", ln=1)
    pdf.set_font("Arial", size=10)
    
    summary_text = generate_explanations(results)
    for line in summary_text.split('\n'):
        pdf.multi_cell(190, 8, txt=line)
        pdf.ln(2)
    
    # Model performance
    pdf.add_page()
    pdf.set_font("Arial", size=14, style='B')
    pdf.cell(200, 10, txt="Model Performance", ln=1)
    pdf.set_font("Arial", size=10)
    
    # Add dataframe as table (first 20 rows)
    col_width = pdf.w / len(df.columns[:5]) - 20
    pdf.set_font("Arial", size=6)
    
    headers = list(df.columns[:5])
    for header in headers:
        pdf.cell(col_width, 8, txt=str(header), border=1)
    pdf.ln()
    
    for i in range(min(20, len(df))):
        for col in headers:
            pdf.cell(col_width, 6, txt=str(df[col].iloc[i])[:30], border=1)
        pdf.ln()
    
    return pdf.output(dest='S').encode('latin1')

def settings_panel():
    """Admin settings panel"""
    
    if st.session_state.user_role != 'admin':
        st.warning("⚠️ You don't have permission to access settings.")
        return
    
    st.markdown("## ⚙️ Admin Settings")
    
    tabs = st.tabs(["Change Password", "Manage Users", "System Info"])
    
    with tabs[0]:
        st.markdown("### 🔒 Change Password")
        
        current = st.text_input("Current Password", type="password")
        new = st.text_input("New Password", type="password")
        confirm = st.text_input("Confirm New Password", type="password")
        
        if st.button("Update Password"):
            if verify_password(st.session_state.username, current):
                if len(new) >= 6:
                    if new == confirm:
                        if change_password(st.session_state.username, new):
                            st.success("✅ Password updated successfully!")
                        else:
                            st.error("❌ Failed to update password")
                    else:
                        st.error("❌ New passwords don't match")
                else:
                    st.error("❌ Password must be at least 6 characters")
            else:
                st.error("❌ Current password is incorrect")
    
    with tabs[1]:
        st.markdown("### 👥 User Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Add New User")
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["viewer", "admin"])
            
            if st.button("Add User"):
                if new_username and new_password:
                    if add_user(new_username, new_password, new_role):
                        st.success(f"✅ User {new_username} added!")
                        st.rerun()
                    else:
                        st.error("❌ User already exists")
        
        with col2:
            st.markdown("#### Existing Users")
            conn = sqlite3.connect('users.db')
            users_df = pd.read_sql_query(
                "SELECT username, role, created_at, last_login FROM users",
                conn
            )
            conn.close()
            
            st.dataframe(users_df, use_container_width=True)
            
            # Delete user
            delete_username = st.selectbox(
                "Delete User (except admin)",
                [u for u in users_df['username'].tolist() if u != 'admin']
            )
            
            if st.button("Delete User", type="secondary"):
                if delete_user(delete_username):
                    st.success(f"✅ User {delete_username} deleted!")
                    st.rerun()
                else:
                    st.error("❌ Failed to delete user")
    
    with tabs[2]:
        st.markdown("### 💻 System Information")
        
        import sys
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("App Version", "2.0.0")
            st.metric("Python Version", sys.version.split()[0])
            st.metric("Streamlit Version", st.__version__)
        
        with col2:
            # Database size
            import os
            if os.path.exists('users.db'):
                size_mb = os.path.getsize('users.db') / (1024 * 1024)
                st.metric("Database Size", f"{size_mb:.2f} MB")
            st.metric("Session ID", str(hash(st.session_state.username))[:8])
        
        if st.button("🗑️ Clear Cache"):
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        if st.button("📋 Export Logs"):
            if os.path.exists('app.log'):
                with open('app.log', 'r') as f:
                    log_content = f.read()
                st.download_button(
                    "Download app.log",
                    log_content,
                    "app.log"
                )

def home_page():
    """Home page content"""
    
    st.markdown("""
    <div class="main-header fade-in">
        <h1>🚀 Pragathi Dynamic Predictive & Inventory Optimization Engine</h1>
        <p>AI-Powered Supply Chain Analytics for Smarter Decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to Pragathi Analysis Engine! 👋
    
    This powerful tool helps you:
    
    📊 **Predict Future Demand** - Using advanced machine learning to forecast your sales or demand
    📦 **Optimize Inventory** - Calculate exactly how much safety stock you need
    🚚 **Prevent Stockouts** - Get automatic alerts when you need to reorder
    💰 **Reduce Costs** - Minimize excess inventory while maintaining service levels
    
    ### How It Works:
    
    1. **Upload your data** (CSV or Excel file)
    2. **Select your target column** (what you want to predict, like "Sales")
    3. **Click Run Optimization** and get instant insights!
    
    ### What You Need:
    
    - Any sales, demand, or inventory data
    - At least one numeric column to predict
    - Optional but helpful: Date column, Stock levels, Lead times
    
    ### Ready to get started?
    
    **📂 Upload your file using the widget below!**
    """)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📁 Drag and drop your CSV or Excel file here, or click to browse",
        type=['csv', 'xlsx'],
        help="Supports CSV and Excel files up to 200MB"
    )
    
    if uploaded_file:
        # Show file info
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.success(f"✅ File uploaded successfully! 📁 {uploaded_file.name} ({file_size_mb:.1f} MB)")
        
        # Load and clean data
        with st.spinner("🧹 Cleaning and preparing your data..."):
            progress_bar = st.progress(0)
            
            def update_progress(progress, message):
                progress_bar.progress(progress)
            
            df, cleaning_summary = load_and_clean_file(uploaded_file, update_progress)
            st.session_state.cleaned_data = df
            st.session_state.cleaning_summary = cleaning_summary
            
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
        
        # Display data summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            if cleaning_summary['date_col_detected']:
                st.metric("Date Column", cleaning_summary['date_col_detected'], "✅ Detected")
            else:
                st.metric("Date Column", "None", "⚠️ Not found")
        with col4:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("Memory Usage", f"{memory_usage:.2f} MB")
        
        # Cleaning summary
        with st.expander("🧹 Data Cleaning Summary", expanded=False):
            st.write(f"✅ Removed {cleaning_summary['empty_rows_removed']} empty rows")
            st.write(f"✅ Removed {cleaning_summary['empty_cols_removed']} empty columns")
            st.write(f"✅ Removed {cleaning_summary['duplicates_removed']} duplicate rows")
            st.write(f"✅ Filled {cleaning_summary['missing_values_filled']} missing values")
            st.write(f"✅ Detected {cleaning_summary['outliers_detected']} outlier values")
            st.write(f"✅ Final dataset: {cleaning_summary['final_rows']} rows × {cleaning_summary['final_cols']} columns")
        
        # Data preview
        with st.expander("👁️ Preview Your Data (click to expand)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Column type table
        st.markdown("### 📊 Column Types Detected")
        
        col_types = []
        date_col = cleaning_summary['date_col_detected']
        inv_cols = cleaning_summary['inventory_cols']
        
        for col in df.columns:
            if col == date_col:
                col_types.append({"Column": col, "Type": "🔵 Date Column"})
            elif col in inv_cols.values():
                col_types.append({"Column": col, "Type": "🟠 Inventory Column"})
            elif df[col].dtype in ['int64', 'float64']:
                col_types.append({"Column": col, "Type": "🟢 Numeric"})
            else:
                col_types.append({"Column": col, "Type": "🟡 Categorical"})
        
        st.dataframe(pd.DataFrame(col_types), use_container_width=True)
        
        # Auto-detect target
        st.markdown("### 🎯 Target Detection")
        
        detected_target, confidence, candidates = auto_detect_target(
            df, date_col, inv_cols
        )
        
        if detected_target:
            if "High" in confidence or "Medium" in confidence:
                st.success(f"🎯 Auto-detected target: **{detected_target}** (Confidence: {confidence})")
                st.markdown(f"👍 We recommend using **{detected_target}** as your target variable.")
            else:
                st.warning(f"⚠️ Auto-detected target: {detected_target} (Confidence: {confidence})")
        else:
            st.info("ℹ️ No clear target detected automatically.")
        
        # Manual target selection
        if st.button("✏️ Change Target"):
            st.session_state.show_target_selector = True
        
        if st.session_state.get('show_target_selector', False) or confidence in ["Low 🟠", "None 🔴"]:
            all_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            target_options = all_numeric if all_numeric else df.columns.tolist()
            
            st.session_state.target_column = st.selectbox(
                "Select your target variable (what you want to predict)",
                target_options,
                index=target_options.index(detected_target) if detected_target in target_options else 0
            )
            
            if st.button("Confirm Target"):
                st.session_state.show_target_selector = False
                st.rerun()
        else:
            st.session_state.target_column = detected_target
        
        # Detect analysis type
        if st.session_state.target_column:
            analysis_type = detect_analysis_type(df, st.session_state.target_column)
            st.session_state.analysis_type = analysis_type
            
            st.info(f"🧠 Analysis Type: **{analysis_type}**")
            
            if analysis_type == "Time Series" and date_col:
                st.success(f"⏰ Using {date_col} for time series analysis")
            
            # Run analysis button
            if st.button("🚀 Run Predictive Optimization", use_container_width=True):
                st.session_state.run_analysis = True
                st.rerun()
    
    else:
        st.info("👈 Please upload your CSV or Excel file to begin.")

def analysis_page():
    """Analysis page with ML results"""
    
    if st.session_state.cleaned_data is None:
        st.warning("Please upload data on the Home page first.")
        return
    
    st.markdown("## 📈 Predictive Analysis Results")
    
    if not st.session_state.get('run_analysis', False):
        st.info("Click 'Run Predictive Optimization' on the Home page to start analysis.")
        return
    
    df = st.session_state.cleaned_data.copy()
    target = st.session_state.target_column
    analysis_type = st.session_state.analysis_type
    date_col = st.session_state.cleaning_summary.get('date_col_detected')
    
    # Progress tracking
    progress_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Step 1: Train models
        progress_placeholder.info("🔧 Training models... (This may take a minute)")
        progress_bar.progress(20)
        
        df, model, model_name, results_df, models = train_models(
            df, target, analysis_type, date_col,
            lambda p, m: progress_bar.progress(int(p * 100))
        )
        
        st.session_state.predictions = df
        st.session_state.best_model = model
        st.session_state.model_name = model_name
        st.session_state.model_results = results_df
        
        progress_bar.progress(60)
        progress_placeholder.info("📊 Generating visualizations...")
        
        # Display model comparison
        if results_df is not None and len(results_df) > 0:
            st.markdown("### 🤖 Model Performance Comparison")
            
            # Sort by appropriate metric
            if analysis_type == 'Regression':
                sort_col = 'MAE'
                best_model_row = results_df.loc[results_df[sort_col].idxmin()]
            else:
                sort_col = 'Accuracy'
                best_model_row = results_df.loc[results_df[sort_col].idxmax()]
            
            # Create model comparison chart
            fig = make_subplots(rows=1, cols=1)
            
            if analysis_type == 'Regression':
                fig.add_trace(go.Bar(
                    x=results_df['Model'][:5],
                    y=results_df['MAE'][:5],
                    name='MAE',
                    marker_color='#ff6b6b'
                ))
                fig.update_layout(yaxis_title="Mean Absolute Error")
            else:
                fig.add_trace(go.Bar(
                    x=results_df['Model'][:5],
                    y=results_df['Accuracy'][:5],
                    name='Accuracy',
                    marker_color='#4CAF50'
                ))
                fig.update_layout(yaxis_title="Accuracy")
            
            fig.update_layout(
                title="Top 5 Models",
                xaxis_title="Model",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show results table
            st.dataframe(results_df, use_container_width=True)
            
            # Highlight best model
            st.success(f"🏆 Best Model: **{model_name}**")
            
            # Calculate metrics for explanations
            if analysis_type == 'Regression':
                predictions = df[f'Predicted_{target}']
                actuals = df[target]
                mae = mean_absolute_error(actuals, predictions)
                r2 = r2_score(actuals, predictions)
                
                results_dict = {
                    'model_name': model_name,
                    'mae': mae,
                    'r2': r2,
                    'target': target
                }
            else:
                results_dict = {
                    'model_name': model_name,
                    'target': target
                }
        else:
            results_dict = {'model_name': 'Baseline', 'target': target}
        
        # Step 2: Create visualizations
        progress_bar.progress(70)
        
        # Actual vs Predicted plot
        if f'Predicted_{target}' in df.columns:
            st.markdown("### 📊 Prediction Visualizations")
            
            fig_actual_vs_pred = create_actual_vs_predicted_plot(
                df, target, f'Predicted_{target}'
            )
            st.plotly_chart(fig_actual_vs_pred, use_container_width=True)
            
            # Residual distribution
            residuals = df[target] - df[f'Predicted_{target}']
            fig_residuals = go.Figure()
            fig_residuals.add_trace(go.Histogram(
                x=residuals,
                nbinsx=30,
                marker_color='#667eea',
                name='Residuals'
            ))
            fig_residuals.update_layout(
                title=f"Residual Distribution (Mean: {residuals.mean():.2f}, Std: {residuals.std():.2f})",
                xaxis_title="Residual (Actual - Predicted)",
                yaxis_title="Frequency",
                height=400
            )
            st.plotly_chart(fig_residuals, use_container_width=True)
            
            # Feature importance if available
            if model and hasattr(model, 'feature_importances_'):
                feature_cols = [col for col in df.columns if col != target and col != f'Predicted_{target}']
                feature_cols = feature_cols[:min(20, len(feature_cols))]
                
                fig_importance = create_feature_importance_plot(model, feature_cols)
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
        
        # Step 3: Inventory optimization
        progress_bar.progress(80)
        
        inv_cols = st.session_state.cleaning_summary.get('inventory_cols', {})
        stock_col = inv_cols.get('stock_col')
        
        if stock_col and target in df.columns:
            st.markdown("### 📦 Inventory Optimization")
            
            df_inventory = calculate_inventory_optimization(
                df, target, inv_cols, date_col, st.session_state.inventory_z_score
            )
            st.session_state.inventory_results = df_inventory
            
            # Inventory summary
            trigger_count = (df_inventory['Procurement_Status'] == "⚠️ TRIGGER SUPPLIER PURCHASE ORDER").sum()
            total_skus = df_inventory[inv_cols.get('sku_col', '_temp')].nunique() if inv_cols.get('sku_col') else 1
            total_order_qty = df_inventory['Recommended_Order_Qty'].sum()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📦 Items Needing Order", f"{trigger_count:,}", 
                         delta=f"{(trigger_count/len(df_inventory)*100):.1f}%")
            with col2:
                st.metric("📊 Total SKUs", f"{total_skus:,}")
            with col3:
                st.metric("🎯 Avg Safety Stock", f"{df_inventory['Safety_Stock'].mean():.0f}")
            with col4:
                st.metric("🚚 Total Order Qty", f"{int(total_order_qty):,}")
            
            # Create inventory visualization
            if inv_cols.get('sku_col'):
                fig_inventory = create_inventory_chart(
                    df_inventory,
                    inv_cols['sku_col'],
                    stock_col,
                    'Reorder_Point'
                )
                if fig_inventory:
                    st.plotly_chart(fig_inventory, use_container_width=True)
            
            # Procurement actions table
            st.markdown("#### 📋 Procurement Actions")
            
            # Highlight rows that need ordering
            def highlight_procurement(row):
                if row['Procurement_Status'] == "⚠️ TRIGGER SUPPLIER PURCHASE ORDER":
                    return ['background-color: #ffcccc'] * len(row)
                return [''] * len(row)
            
            display_cols = ['Procurement_Status', 'Safety_Stock', 'Reorder_Point', 'Recommended_Order_Qty']
            if inv_cols.get('sku_col'):
                display_cols = [inv_cols['sku_col']] + display_cols
            
            display_cols = [col for col in display_cols if col in df_inventory.columns]
            
            styled_df = df_inventory[display_cols].head(20).style.apply(highlight_procurement, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Update results dict
            results_dict['inventory_summary'] = {
                'trigger_count': trigger_count,
                'total_skus': total_skus,
                'total_order_qty': total_order_qty,
                'top_skus': df_inventory[df_inventory['Recommended_Order_Qty'] > 0][
                    inv_cols.get('sku_col', target)
                ].head(5).tolist() if inv_cols.get('sku_col') else []
            }
        
        # Step 4: Generate explanations
        progress_bar.progress(90)
        
        # Calculate trend
        if len(df) > 1 and target in df.columns:
            trend = (df[target].iloc[-1] - df[target].iloc[0]) / df[target].iloc[0] if df[target].iloc[0] != 0 else 0
            results_dict['trend'] = trend
        
        explanations = generate_explanations(results_dict)
        
        st.markdown("### 📋 What This Means For You")
        st.markdown(f'<div class="success-card">{explanations}</div>', unsafe_allow_html=True)
        
        progress_bar.progress(100)
        progress_placeholder.success("✅ Analysis complete!")
        
    except Exception as e:
        progress_placeholder.error(f"❌ Analysis failed: {str(e)}")
        logging.error(f"Analysis error: {traceback.format_exc()}")
        st.error(f"""
        **Something went wrong while analyzing your data.**
        
        **Error:** {str(e)}
        
        **Possible solutions:**
        - Make sure your target column has enough variation
        - Try selecting a different target column
        - Check if your data has at least 10 rows
        - Ensure numeric columns don't have too many missing values
        """)

def inventory_page():
    """Inventory optimization page"""
    
    if st.session_state.inventory_results is not None:
        df = st.session_state.inventory_results
        
        st.markdown("## 📊 Inventory Optimization Details")
        
        # Filter controls
        st.markdown("### 🔍 Filter Inventory Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            status_filter = st.multiselect(
                "Procurement Status",
                options=df['Procurement_Status'].unique(),
                default=df['Procurement_Status'].unique()
            )
        
        with col2:
            min_order = st.slider(
                "Minimum Order Quantity",
                min_value=0,
                max_value=int(df['Recommended_Order_Qty'].max()),
                value=0
            )
        
        # Apply filters
        filtered_df = df[df['Procurement_Status'].isin(status_filter)]
        filtered_df = filtered_df[filtered_df['Recommended_Order_Qty'] >= min_order]
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Filtered SKUs", len(filtered_df))
        with col2:
            st.metric("Total Order Quantity", f"{int(filtered_df['Recommended_Order_Qty'].sum()):,}")
        with col3:
            st.metric("Avg Safety Stock", f"{filtered_df['Safety_Stock'].mean():.0f}")
        with col4:
            avg_volatility = filtered_df['Demand_Volatility'].mean()
            st.metric("Avg Demand Volatility", f"{avg_volatility:.2f}")
        
        # Interactive Z-score adjustment
        st.markdown("### ⚙️ Adjust Service Level")
        
        new_z = st.slider(
            "Service Level Z-Score",
            min_value=0.50,
            max_value=3.00,
            step=0.05,
            value=st.session_state.inventory_z_score,
            help="Higher Z-score = more safety stock = lower stockout risk"
        )
        
        if new_z != st.session_state.inventory_z_score:
            st.session_state.inventory_z_score = new_z
            st.rerun()
        
        # Procurement table with highlighting
        st.markdown("### 📋 Complete Procurement Plan")
        
        def highlight_needed(row):
            if row['Procurement_Status'] == "⚠️ TRIGGER SUPPLIER PURCHASE ORDER":
                return ['background-color: #ffcccc'] * len(row)
            return [''] * len(row)
        
        st.dataframe(
            filtered_df.style.apply(highlight_needed, axis=1),
            use_container_width=True,
            height=400
        )
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv(filtered_df)
            st.download_button(
                "📥 Download Filtered Results (CSV)",
                csv_data,
                f"inventory_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            if FPDF_AVAILABLE:
                pdf_data = export_to_pdf(filtered_df, {})
                if pdf_data:
                    st.download_button(
                        "📄 Download PDF Report",
                        pdf_data,
                        f"inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        use_container_width=True
                    )
            else:
                st.info("📦 Install fpdf2 for PDF export: `pip install fpdf2`")
    
    else:
        st.info("ℹ️ Run the analysis first to see inventory optimization results.")

def export_page():
    """Export results page"""
    
    st.markdown("## 📥 Export Results")
    
    if st.session_state.predictions is not None:
        df = st.session_state.predictions
        
        st.markdown("### Available Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv_data = export_to_csv(df)
            st.download_button(
                "📥 Download Complete Results (CSV)",
                csv_data,
                f"optimized_procurement_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            # Excel Export
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                # Write multiple sheets
                df.to_excel(writer, sheet_name='Predictions', index=False)
                
                if st.session_state.inventory_results is not None:
                    st.session_state.inventory_results.to_excel(writer, sheet_name='Inventory', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Rows', 'Target Column', 'Model Used', 'Analysis Date'],
                    'Value': [
                        len(df),
                        st.session_state.target_column,
                        st.session_state.get('model_name', 'N/A'),
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                "📊 Download Excel Report",
                excel_buffer.getvalue(),
                f"pragathi_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
        
        with col3:
            # JSON Export
            json_data = df.to_json(orient='records', date_format='iso')
            st.download_button(
                "📋 Download JSON Data",
                json_data,
                f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )
        
        # Sample preview
        st.markdown("### Preview of Export Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.info(f"Total rows to export: {len(df):,}")
        
    else:
        st.info("ℹ️ No analysis results available. Please run the analysis first.")

def main():
    """Main application entry point"""
    
    # Check authentication
    if not st.session_state.authenticated:
        login_ui()
        return
    
    # Check session timeout (60 minutes)
    if st.session_state.login_time:
        time_elapsed = (datetime.now() - st.session_state.login_time).total_seconds() / 60
        if time_elapsed > 60:
            st.session_state.authenticated = False
            st.warning("Session expired. Please login again.")
            st.rerun()
    
    # Sidebar navigation
    sidebar_navigation()
    
    # Main content based on navigation
    if st.session_state.sidebar_page == "🏠 Home":
        home_page()
    elif st.session_state.sidebar_page == "📈 Analysis":
        analysis_page()
    elif st.session_state.sidebar_page == "📊 Inventory":
        inventory_page()
    elif st.session_state.sidebar_page == "📄 Export":
        export_page()
    elif st.session_state.sidebar_page == "⚙️ Settings":
        settings_panel()

if __name__ == "__main__":
    main()
