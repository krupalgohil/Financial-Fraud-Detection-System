"""
Simple AML Fraud Detection Dashboard
=====================================
Fresher-level Streamlit app — clean and simple.
Run: streamlit run streamlit_app_fresher.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, roc_curve, precision_score, recall_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AML Fraud Detector", page_icon="💳", layout="wide")

st.title("💳 AML Fraud Detection Dashboard")
st.markdown("**Built by:** Your Name &nbsp;|&nbsp; **Dataset:** PaySim &nbsp;|&nbsp; **Model:** XGBoost")
st.markdown("---")

with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload PaySim CSV", type=['csv'])
    sample_size   = st.slider("Sample Size", 5000, 30000, 10000, 1000)
    threshold     = st.slider("Alert Threshold", 0.1, 0.9, 0.5, 0.05)

@st.cache_data(show_spinner=False)
def load_data(file, n):
    if file is not None:
        df = pd.read_csv(file)
        return df.sample(min(n, len(df)), random_state=42).reset_index(drop=True), True
    np.random.seed(42)
    nf = int(n * 0.013); nl = n - nf
    legit = pd.DataFrame({
        'step': np.random.randint(1,744,nl),
        'type': np.random.choice(['PAYMENT','CASH-IN','DEBIT','CASH-OUT'],nl),
        'amount': np.random.exponential(50000,nl).clip(1,500000),
        'oldbalanceOrg': np.random.exponential(80000,nl).clip(0),
        'oldbalanceDest': np.random.exponential(60000,nl).clip(0),
        'isFraud': 0
    })
    legit['newbalanceOrig'] = (legit['oldbalanceOrg'] - legit['amount']).clip(0)
    legit['newbalanceDest'] = legit['oldbalanceDest'] + legit['amount']
    fraud = pd.DataFrame({
        'step': np.random.randint(1,744,nf),
        'type': np.random.choice(['TRANSFER','CASH-OUT'],nf),
        'amount': np.random.uniform(100000,900000,nf),
        'oldbalanceOrg': np.random.uniform(100000,900000,nf),
        'newbalanceOrig': np.zeros(nf),
        'oldbalanceDest': np.zeros(nf),
        'isFraud': 1
    })
    fraud['amount'] = fraud['oldbalanceOrg']
    fraud['newbalanceDest'] = fraud['amount']
    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    df['nameOrig'] = 'C' + pd.Series(np.random.randint(100000000,999999999,len(df))).astype(str)
    df['nameDest']  = 'C' + pd.Series(np.random.randint(100000000,999999999,len(df))).astype(str)
    return df.reset_index(drop=True), False

@st.cache_data(show_spinner=False)
def add_features(_df):
    df = _df.copy()
    df['account_drained']    = ((df['oldbalanceOrg']>0) & (df['newbalanceOrig']==0)).astype(int)
    df['amount_to_balance']  = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['balance_change']     = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['is_high_risk_type']  = df['type'].isin(['TRANSFER','CASH-OUT','CASH_OUT']).astype(int)
    df['hour_of_day']        = df['step'] % 24
    df['is_late_night']      = ((df['hour_of_day']>=22)|(df['hour_of_day']<=4)).astype(int)
    return df

@st.cache_resource(show_spinner=False)
def train_model(_df):
    features = ['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest',
                'newbalanceDest','account_drained','amount_to_balance',
                'balance_change','is_high_risk_type','hour_of_day','is_late_night']
    features = [f for f in features if f in _df.columns]
    X = _df[features].fillna(0); y = _df['isFraud']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale_w = (y_train==0).sum() / y_train.sum()
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                           scale_pos_weight=scale_w, use_label_encoder=False,
                           eval_metric='logloss', random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, X_test, y_test, features

with st.spinner("Loading data and training model..."):
    df_raw, is_real = load_data(uploaded_file, sample_size)
    df_fe = add_features(df_raw)
    model, X_test, y_test, FEATURES = train_model(df_fe)

yp = model.predict_proba(X_test)[:,1]
ypred = (yp >= threshold).astype(int)
roc = roc_auc_score(y_test, yp)
f1_ = f1_score(y_test, ypred)

if not is_real:
    st.info("💡 Demo mode — upload your PaySim CSV from Kaggle for real analysis.")

c1,c2,c3,c4,c5 = st.columns(5)
c1.metric("Transactions", f"{len(df_raw):,}")
c2.metric("Fraud Cases",  f"{int(df_raw['isFraud'].sum()):,}")
c3.metric("ROC-AUC",      f"{roc:.4f}")
c4.metric("F1 Score",     f"{f1_:.4f}")
c5.metric("Alerts",       f"{int(ypred.sum()):,}")
st.markdown("---")

tab1,tab2,tab3,tab4 = st.tabs(["📊 EDA","🤖 Model","🧠 Features","🔍 Score"])

with tab1:
    st.subheader("Exploratory Data Analysis")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("**Fraud Rate by Type**")
        td = df_raw.groupby('type')['isFraud'].mean().reset_index()
        fig,ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=td, x='type', y='isFraud', palette='Reds_r', ax=ax)
        ax.set_ylabel('Fraud Rate'); ax.set_xlabel(''); plt.xticks(rotation=15)
        plt.tight_layout(); st.pyplot(fig)
        st.info("💡 Fraud only in TRANSFER & CASH-OUT — monitor only these 2 types!")

    with c2:
        st.markdown("**Amount by Class**")
        df_raw['label'] = df_raw['isFraud'].map({0:'Legitimate',1:'Fraud'})
        clip = float(df_raw['amount'].quantile(0.99))
        fig,ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=df_raw[df_raw['amount']<=clip], x='label', y='amount',
                    palette={'Legitimate':'#2A9D8F','Fraud':'#E63946'}, ax=ax)
        ax.set_xlabel(''); plt.tight_layout(); st.pyplot(fig)
        st.info("💡 Fraudulent transactions are much larger on average.")

with tab2:
    st.subheader("Model Performance")
    c1,c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_test, ypred)
        fig,ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred Legit','Pred Fraud'],
                    yticklabels=['Actual Legit','Actual Fraud'],
                    linewidths=2, linecolor='white', annot_kws={'size':14,'weight':'bold'})
        plt.tight_layout(); st.pyplot(fig)
    with c2:
        fpr,tpr,_ = roc_curve(y_test, yp)
        fig,ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr,tpr,color='#E63946',linewidth=2.5,label=f'AUC={roc:.4f}')
        ax.plot([0,1],[0,1],'k--',alpha=0.4,label='Random')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig)

with tab3:
    st.subheader("Feature Importance")
    fi = pd.DataFrame({'feature':FEATURES,'importance':model.feature_importances_}).sort_values('importance',ascending=True)
    fig,ax = plt.subplots(figsize=(8,5))
    sns.barplot(data=fi, x='importance', y='feature', palette='Reds_r', ax=ax)
    ax.set_xlabel('Importance'); ax.set_ylabel('')
    plt.tight_layout(); st.pyplot(fig)
    st.dataframe(pd.DataFrame({
        'Feature':['account_drained','amount_to_balance','is_high_risk_type','is_late_night'],
        'Meaning':['Account emptied to $0','Fraction of balance transferred','TRANSFER or CASH-OUT type','Late night (22:00–04:00)']
    }), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Score a Transaction")
    c1,c2 = st.columns(2)
    col_b1, col_b2 = st.columns(2)
    fill_f = col_b1.button("🚨 Load Fraud Example")
    fill_l = col_b2.button("✅ Load Legit Example")
    d = dict(s=412,t='TRANSFER',a=850000.0,obO=850000.0,nbO=0.0,obD=0.0,nbD=850000.0) if fill_f else \
        dict(s=100,t='PAYMENT', a=250.0,   obO=5000.0,  nbO=4750.0,obD=0.0,nbD=250.0)  if fill_l else \
        dict(s=100,t='PAYMENT', a=500.0,   obO=10000.0, nbO=9500.0,obD=5000.0,nbD=5500.0)
    with c1:
        si  = st.number_input("Step",1,744,d['s'])
        tx  = st.selectbox("Type",['PAYMENT','TRANSFER','CASH-OUT','CASH-IN','DEBIT'],
                            index=['PAYMENT','TRANSFER','CASH-OUT','CASH-IN','DEBIT'].index(d['t']))
        am  = st.number_input("Amount ($)",0.0,1e7,float(d['a']))
        obO = st.number_input("Sender Old Balance ($)",0.0,1e7,float(d['obO']))
    with c2:
        nbO = st.number_input("Sender New Balance ($)",0.0,1e7,float(d['nbO']))
        obD = st.number_input("Receiver Old Balance ($)",0.0,1e7,float(d['obD']))
        nbD = st.number_input("Receiver New Balance ($)",0.0,1e7,float(d['nbD']))

    if st.button("🔍 Check This Transaction", type="primary", use_container_width=True):
        row = pd.DataFrame([{'step':si,'type':tx,'amount':am,'nameOrig':'C1','nameDest':'C2',
                              'oldbalanceOrg':obO,'newbalanceOrig':nbO,'oldbalanceDest':obD,
                              'newbalanceDest':nbD,'isFraud':0}])
        rfe = add_features(row)
        for col in FEATURES:
            if col not in rfe.columns: rfe[col]=0
        prob = model.predict_proba(rfe[FEATURES].fillna(0))[0][1]
        is_f = prob >= threshold
        r1,r2 = st.columns([1,2])
        with r1:
            if is_f: st.error(f"🚨 FRAUD ALERT\n\nProbability: **{prob:.1%}**")
            else:    st.success(f"✅ LEGITIMATE\n\nProbability: **{prob:.1%}**")
            st.progress(float(prob))
        with r2:
            st.markdown("**Risk Factors:**")
            for triggered, desc in [
                (obO>0 and nbO==0, "Account drained to $0"),
                (am/(obO+1)>0.9,   "Transferred >90% of balance"),
                (tx in ['TRANSFER','CASH-OUT'], "High-risk type"),
                (am>100000,        "Large amount (>$100K)"),
                (si%24>=22 or si%24<=4, "Late-night activity"),
            ]:
                st.markdown(f"{'🔴 **TRIGGERED**' if triggered else '🟢 Clear'} — {desc}")

st.markdown("---")
st.caption("💳 AML Fraud Detection | Python · Seaborn · XGBoost · Streamlit | Dataset: PaySim (Kaggle)")
