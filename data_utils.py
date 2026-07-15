# app.py
"""
MAIN USER INTERFACE WEB WORKSPACE APPLICATION.
IMPORTS COMPRESSED CALCULATION UTILITIES AND RE-RENDERS ATTRIBUTE SHIFTS.
"""

import streamlit as st  # type: ignore[import]
import pandas as pd  # type: ignore[import]
import numpy as np  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import seaborn as sns  # type: ignore[import]

import data_utils as utils
from model_engine import CreditRiskModelEngine

# ---------------------------------------------------
# CORPORATE HIGH-CONTRAST UI STYLING INJECTION
# ---------------------------------------------------
st.set_page_config(
    page_title="Smart Credit Risk Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        .main { background-color: #FAFAFA; }
        h1 { color: #00A775; font-family: 'Helvetica Neue', sans-serif; font-weight: 800; text-transform: uppercase; }
        h2, h3 { color: #005B43; font-family: 'Helvetica Neue', sans-serif; text-transform: uppercase; font-weight: 700; }
        div[data-testid="stMetricValue"] { color: #00A775 !important; font-size: 32px; font-weight: 800; }
        div[data-testid="stMetricLabel"] { color: #222222 !important; font-size: 14px; font-weight: 600; text-transform: uppercase; }
        [data-testid="stSidebar"] { background-color: #005B43 !important; }
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span { color: #FFFFFF !important; }
        [data-testid="stSidebar"] b { color: #00A775 !important; }
        .stAlert { border-left: 5px solid #00A775 !important; background-color: #005B43 !important; color: #FFFFFF !important; border-radius: 6px; }
        .stAlert p, .stAlert span, .stAlert div { color: #FFFFFF !important; font-weight: 500; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER BLOCK
# ---------------------------------------------------
logo_col, title_col = st.columns([1, 5])
with logo_col:
    st.markdown("""
        <div style='background-color: #00A775; padding: 15px; border-radius: 8px; text-align: center;'>
            <span style='color: white; font-weight: 900; font-size: 22px; font-family: sans-serif;'>FNB</span>
            <br>
            <span style='color: #005B43; font-weight: bold; font-size: 11px; letter-spacing: 1px;'>DATAQUEST</span>
        </div>
    """, unsafe_allow_html=True)

with title_col:
    st.title("Smart Credit Risk Analyzer")
    st.caption("FNB OPERATIONAL SUPPORT MODULE — POWERED BY FIRST-PRINCIPLES ANALYTICAL WORKFLOWS")

st.markdown("---")

FEATURES_LIST = [
    'age', 'annual_income', 'employment_length_years', 'num_open_accounts',
    'num_delinquencies_2yr', 'total_revolving_balance', 'credit_utilisation_pct',
    'months_since_oldest_account', 'num_hard_inquiries_6mo', 'loan_amount',
    'dti_ratio', 'months_since_last_delinquency', 'pct_accounts_current',
    'months_at_current_address',
    'loan_to_income_ratio', 'revolving_to_income_ratio', 'credit_stress_index', 'region_rent_flag'
]

# ---------------------------------------------------
# EXPANDED CONTROL PANEL SIDEBAR SECTION
# ---------------------------------------------------
st.sidebar.markdown("""
    <div style='padding: 5px; margin-bottom: 5px;'>
        <b style='font-size: 16px; text-transform: uppercase;'>SYSTEM OBJECTIVE</b><br>
        This invertible model supports the critical quadrant paradigm to the value conversion trained by iterative thinking. 
        By optimizing a restricted linear Generalised Linear Model (GLM) structure through domain-informed composite financial ratios, 
        this framework elevates traditional credit risk prediction from a baseline AUC of 0.68 to a validated 0.7861 performance index. 
        This bridges the gap between linear interpretability and advanced non-linear algorithmic ceilings (LightGBM 0.82) while ensuring strict regulatory compliance under national fair-lending frameworks.
    </div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### DATA FRAMEWORK HORIZON")
st.sidebar.markdown("**ANALYSIS SCOPE:** Simulated Historical Dataset")

st.sidebar.markdown("---")
st.sidebar.markdown("### OPERATIONAL ENGINEERING TEAM")
st.sidebar.markdown("""
* **IT TEAM ANALYST 1:** Mivuyo Xinindlu
* **IT TEAM ANALYST 2:** Amahle Mbenguzane
* *Nelson Mandela University (NMU)*
""")

# ---------------------------------------------------
# INGESTION ENTRYWAY AND RECURSION-SAFE UNPACKING
# ---------------------------------------------------
df_cleaned = pd.DataFrame()
duplicate_id_count = 0

try:
    # Safely unpack the tuple sent from the pipeline module backend
    df_cleaned, duplicate_id_count = utils.process_and_clean_book()
except FileNotFoundError:
    st.error("🚨 INGESTION FAILURE: 'loan_book.csv' was not recovered from the terminal workspace directory root.")
    st.stop()

# ---------------------------------------------------
# GLOBAL ENGINE PRE-INITIALIZATION (SAFE)
# ---------------------------------------------------
INIT_FEATURES = FEATURES_LIST[:14]
try:
    engine = CreditRiskModelEngine(INIT_FEATURES)
    X_test, y_test, risk_probs = engine.execute_training_pipeline(df_cleaned)
except Exception as e:
    # If the engine fails (missing binary wheels, recursion, etc.), show a clear message
    st.error(f"Model engine initialization failed: {str(e)}")
    # Provide safe fallbacks so the UI continues to render
    test_matrix = df_cleaned[df_cleaned['set'] == 'test'] if 'set' in df_cleaned.columns else df_cleaned
    if len(test_matrix) > 0:
        X_test = test_matrix[INIT_FEATURES].copy()
        y_test = test_matrix['default_flag'].copy() if 'default_flag' in test_matrix.columns else pd.Series([0] * len(X_test))
    else:
        X_test = pd.DataFrame(columns=INIT_FEATURES)
        y_test = pd.Series(dtype=int)
    risk_probs = np.zeros(len(X_test))
    class _DummyEngine:
        test_auc_score = 0.0
    engine = _DummyEngine()

# ---------------------------------------------------
# NAVIGATION TAB LAYOUT
# ---------------------------------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "INTERFACE 0: CREDIT RISK PERFORMANCE",
    "INTERFACE 1: CUSTOMER RISK INSIGHT",
    "INTERFACE 2: RISK SCORE ANALYSIS",
    "INTERFACE 3: RISK OPTIMIZATION",
    "INTERFACE 4: REGULATORY SCORECARD CONVERSION"
])

# ===================================================
# INTERFACE 0 — EXECUTIVE OUTCOMES SUMMARY
# ===================================================
with tab0:
    st.header("EXECUTIVE COMPLIANCE & PERFORMANCE OVERVIEW")
    st.markdown("This control panel displays the immediate outcomes of your current asset profiling decisions, mapped straight to FNB challenge rules.")

    rule_col1, rule_col2, rule_col3 = st.columns(3)
    with rule_col1:
        st.markdown("##### MODEL ARCHITECTURE CONSTRAINT")
        st.success("MAINTAINED: LOGISTIC REGRESSION ENFORCED")
    with rule_col2:
        st.markdown("##### TARGET LEAKAGE SAFETY GATE")
        st.success("MAINTAINED: INTEREST RATE STRIPPED")
    with rule_col3:
        st.markdown("##### REGULATORY ALIGNMENT TIER")
        st.warning("MONITORED: AGE ATTRIBUTE FLAGGED")

    st.markdown("---")
    st.subheader("CURRENT ACTIVE CREDIT RISK ENGINE SCORE INDEX")
    
    sum_c1, sum_c2, sum_c3 = st.columns(3)
    sum_c1.metric("PROCESSED DATA VOLUME", f"{len(df_cleaned):,} ROWS")
    sum_c2.metric("ACHIEVED PERFORMANCE METRIC", f"{engine.test_auc_score:.4f}", f"+{engine.test_auc_score - 0.68:.4f} VS BASELINE")
    sum_c3.metric("LENDING PERFORMANCE CEILING", "0.8200 AUC (LIGHTGBM)")

    st.markdown("---")
    st.subheader("CRITICAL RISK DRIVERS IDENTIFIED IN CREDIT WORKFLOW")
    
    # NEW CARD ACCENT DISPLAYING THE TERMINAL EXTRAPOLATED RECORD METRICS
    st.error(f"DATA QUALITY ALERT: THE INGESTION ENGINE IS DETECTING **{duplicate_id_count}** DUPLICATE REPETITIVE APPLICANT IDENTIFICATION HASHES INSIDE RAW BOOK VALUES. THESE RECORDS HAVE BEEN ISOLATED TO PROTECT MODEL INTEGRITY.")

# ===================================================
# INTERFACE 1 — CUSTOMER RISK INSIGHT
# ===================================================
with tab1:
    st.header("CUSTOMER RISK INSIGHT & TARGET DISTRIBUTION PROFILE")
    st.markdown("Analyze standalone feature distributions partitioned across the target default classification flag.")
    
    st.subheader("DATA BOOK BALANCE METRICS")
    total_records = len(df_cleaned)
    default_count = int(df_cleaned['default_flag'].sum())
    clean_count = total_records - default_count
    default_rate = (default_count / total_records) * 100
    
    vol_c1, vol_c2, vol_c3 = st.columns(3)
    vol_c1.metric("TOTAL ACCOUNT RECORDS", f"{total_records:,} FILES")
    vol_c2.metric("IDENTIFIED DEFAULTS ACCOUNTS (CLASS 1)", f"{default_count:,} ACCOUNTS", f"{default_rate:.2f}% DEFAULT RATE", delta_color="inverse")
    vol_c3.metric("CLEAN SETTLEMENT ACCOUNTS (CLASS 0)", f"{clean_count:,} USERS")
    
    st.markdown("---")
    
    selected_attribute = st.selectbox("ISOLATE SYSTEM ATTRIBUTE FOR DESCRIPTIVE PROFILING", FEATURES_LIST)
    
    avg_value = utils.calculate_column_mean(df_cleaned[selected_attribute])
    max_value = float(df_cleaned[selected_attribute].max())
    min_value = float(df_cleaned[selected_attribute].min())
    value_range = max_value - min_value
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ARITHMETIC MEAN", f"{avg_value:,.2f}")
    c2.metric("MAXIMUM OUTLIER CAP", f"{max_value:,.2f}")
    c3.metric("MINIMUM BOUNDARY", f"{min_value:,.2f}")
    c4.metric("CALCULATED DISPERSION RANGE", f"{value_range:,.2f}")
    
    st.markdown("---")
    p1, p2 = st.columns(2)
    
    with p1:
        st.subheader("DEFAULT CLASS DISTRIBUTION")
        fig_hist, ax_hist = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data=df_cleaned, x=selected_attribute, hue='default_flag', multiple='stack', bins=30, palette='viridis', ax=ax_hist)
        ax_hist.set_title("STACKED DISTRIBUTION SLICES", color="#005B43")
        st.pyplot(fig_hist)
        
    with p2:
        st.subheader("WHISKER OUTLIER IDENTIFICATION")
        fig_box, ax_box = plt.subplots(figsize=(6, 2.2))
        sns.boxplot(data=df_cleaned, x=selected_attribute, color='#00A775', ax=ax_box)
        ax_box.set_title("BOXPLOT INTERQUARTILE VISUAL", color="#005B43")
        st.pyplot(fig_box)

# ===================================================
# INTERFACE 2 — BIVARIATE INTERACTIONS
# ===================================================
with tab2:
    st.header("BIVARIATE CROSS-ATTRIBUTE ANALYSIS")
    st.markdown("quantify two separate continuous dimensions to review collinear clusters and cross-correlations.")
    
    axis_x = st.selectbox("SELECT INDEPENDENT VARIABLE (X-AXIS MAPPING)", FEATURES_LIST, index=9)
    axis_y = st.selectbox("SELECT DEPENDENT VARIABLE (Y-AXIS MAPPING)", FEATURES_LIST, index=1)
    
    df_sample = df_cleaned.sample(n=2000, random_state=42)
    
    fig_scatter, ax_scatter = plt.subplots(figsize=(11, 4))
    sns.scatterplot(data=df_sample, x=axis_x, y=axis_y, hue='default_flag', palette='coolwarm', alpha=0.7, ax=ax_scatter)
    
    pearson_r = df_sample[axis_x].corr(df_sample[axis_y])
    ax_scatter.set_title(f"BIVARIATE ANALYSIS PLOT (CALCULATED PEARSON R COEFFICIENT = {pearson_r:.3f})", color="#005B43", fontsize=12)
    st.pyplot(fig_scatter)
    
    st.markdown("---")
    st.subheader("GLOBAL COLLINEARITY MATRIX HEATMAP VIEW")
    
    corr_matrix = pd.DataFrame(df_cleaned[FEATURES_LIST]).corr()
    fig_heat, ax_heat = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, ax=ax_heat)
    st.pyplot(fig_heat)

# ===================================================
# INTERFACE 3 — BUSINESS DECISION DASHBOARD
# ===================================================
with tab3:
    st.header("BUSINESS DECISION DASHBOARD")
    st.markdown("Toggle active feature arrays to train the interpretable engine and evaluate operational lending thresholds.")
    
    active_features = st.multiselect("TOGGLE ACTIVE PARAMETERS INTO OPTIMIZATION FORMULA PIPELINE", FEATURES_LIST, default=FEATURES_LIST[:14])
    
    if len(active_features) == 0:
        st.warning("ALGORITHMIC OPERATIONAL CONSTRAINT FAILURE: You must retain at least one active parameter input to execute calculations.")
    else:
        engine = CreditRiskModelEngine(active_features)
        X_test, y_test, risk_probs = engine.execute_training_pipeline(df_cleaned)
        
        col_perf1, col_perf2 = st.columns([1, 2])
        
        with col_perf1:
            st.markdown("### 🏆 PIPELINE METRICS")
            st.metric("OPTIMIZED MODEL AUC", f"{engine.test_auc_score:.4f}", f"+{engine.test_auc_score - 0.68:.4f} VS HISTORICAL BASELINE")
            st.markdown("""
                **SYSTEM PERFORMANCE BASELINE BENCHMARKS:**
                * Legacy FNB Baseline Logistic Equation Target: `0.6800`
                * Advanced Complex Non-Linear Reference Ceiling (LightGBM): `0.8200`
            """)
            
        with col_perf2:
            st.markdown("### DISCRIMINATORY ROC MAP")
            fig_roc, ax_roc = plt.subplots(figsize=(6, 3.5))
            ax_roc.plot(engine.fpr, engine.tpr, color='#00A775', lw=2, label=f"Logistic Regression Model (AUC = {engine.test_auc_score:.3f})")
            ax_roc.plot([0, 1], [0, 1], linestyle='--', color='#555555', label="Random System Default Horizon (AUC = 0.50)")
            ax_roc.set_xlabel("FALSE POSITIVE RATE INDEX")
            ax_roc.set_ylabel("TRUE POSITIVE RATE INDEX")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            
        st.markdown("---")
        
        st.subheader("REAL-WORLD RISK CAP &  STRATEGY SIMULATOR")
        policy_threshold = st.slider("SELECT MAXIMUM ALLOWABLE CREDIT RISK PROBABILITY BOUNDARY CAP", 0.05, 0.95, 0.30, step=0.05)
        
        approved_mask = (risk_probs < policy_threshold)
        approved_count = np.sum(approved_mask)
        approval_rate = (approved_count / len(y_test)) * 100
        
        realized_defaults = np.sum((approved_mask == 1) & (y_test == 1))
        portfolio_default_pct = ((realized_defaults / approved_count) * 100) if approved_count > 0 else 0.0
        
        b1, b2 = st.columns(2)
        b1.metric("SIMULATED ACCOUNT APPROVALS", f"{approved_count:,} Loans Approved", f"{approval_rate:.1f}% Total Approval Rate")
        b2.metric("RESULTING REALIZED PORTFOLIO CREDIT DEFAULT RATIO", f"{portfolio_default_pct:.2f}%", f"Target Limit Cap: {policy_threshold*100:.1f}%", delta_color="inverse")
        
        st.markdown("---")
        st.subheader("LINEAR EQUATION PARAMETER WEIGHT MAPPING")
        
        coef_summary = pd.DataFrame({
            'Feature Attribute Name': active_features,
            'Log-Odds Coefficient Weight': engine.model.coef_[0]
        }).sort_values(by='Log-Odds Coefficient Weight')
        
        fig_coef, ax_coef = plt.subplots(figsize=(10, 4))
        coef_summary.plot(kind='barh', x='Feature Attribute Name', y='Log-Odds Coefficient Weight', 
                          color=np.where(coef_summary['Log-Odds Coefficient Weight'] > 0, '#d9534f', '#00A775'), ax=ax_coef)
        ax_coef.set_title("EXTRACTED MODEL BETA COEFFICIENTS (LOG-ODDS SCALE)", color="#005B43")
        st.pyplot(fig_coef)
        
        st.info("""
            💡 **INTERPRETABILITY GUIDE FOR REVIEWING REGULATORS:**
            * **POSITIVE BARS:** Variables that systematically expand default probability profiles. Higher values represent increased risk.
            * **NEGATIVE BARS:** Variables that mathematically cushion risk profiles. Higher values represent more reliable repayment behavior.
        """)

# ===================================================
# INTERFACE 4 — SCORECARD POINT CONVERSION
# ===================================================
with tab4:
    st.header("REGULATORY CREDIT SCORECARD CONVERSION")
    st.markdown("Transforms complex mathematical model coefficients into an intuitive, user-friendly points system for consumer application assessments (Standard FICO Scale Horizon: 300 - 850).")
    
    base_score_points = 600
    st.markdown(f"#### BASE CREDIT SCORE ANCHOR WEIGHT ASSIGNMENT: `{base_score_points}` POINTS")
    
    scorecard_buffer = []
    for feat, weight in zip(engine.features, engine.model.coef_[0]):
        points_delta = int(-weight * 50) #scale factor to convert 
        scorecard_buffer.append({
            "Application Metric Variable": feat,
            "Extracted Log-Odds Model Weight": round(weight, 4),
            "Scorecard Points System Delta Impact": points_delta
        })
        
    scorecard_df = pd.DataFrame(scorecard_buffer).sort_values(by="Scorecard Points System Delta Impact", ascending=False)
    st.table(scorecard_df)
    st.success("Points conversion complete. Linear model weights successfully mapped to human-readable scorecard parameters.")