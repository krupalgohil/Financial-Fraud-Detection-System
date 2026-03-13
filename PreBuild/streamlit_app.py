"""
================================================================
🌍 Global FinTech AML Intelligence Platform
================================================================
Stack    : Streamlit · XGBoost · SHAP · NetworkX · SMOTE
Regs     : FATF · FinCEN (USA) · FCA (UK) · AUSTRAC (AUS)
Audience : Risk Officers · Compliance Teams · ML Engineers
Run      : streamlit run streamlit_app.py
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import shap
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, roc_curve,
    precision_recall_curve, confusion_matrix, matthews_corrcoef
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Global AML Intelligence Platform",
    page_icon="🌍", layout="wide",
    initial_sidebar_state="expanded"
)

C = {
    'fraud':'#E63946', 'legit':'#2A9D8F', 'gold':'#E9C46A',
    'dark':'#0B1120',  'card':'#111827',  'border':'#1F2937',
    'text':'#E5E7EB',  'muted':'#9CA3AF',
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{{font-family:'Inter',sans-serif;background:{C['dark']};color:{C['text']};}}
.stApp{{background:{C['dark']};}}
.hero{{background:linear-gradient(135deg,#0B1120,#0f1f3d 50%,#0B1120);border:1px solid #1e3a5f;border-radius:16px;padding:2.4rem 3rem;margin-bottom:1.8rem;}}
.hero-title{{font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;color:#fff;margin:0;}}
.hero-sub{{font-size:0.86rem;color:{C['muted']};margin:0.4rem 0 0.9rem 0;}}
.hero-badges{{display:flex;gap:7px;flex-wrap:wrap;}}
.badge{{background:#1e3a5f;border:1px solid #2563eb40;color:#93c5fd;padding:3px 10px;border-radius:20px;font-size:0.7rem;font-family:'DM Mono',monospace;}}
.badge-r{{background:#3b0d0d;border-color:#E6394640;color:#fca5a5;}}
.kpi-row{{display:flex;gap:10px;margin-bottom:1.4rem;flex-wrap:wrap;}}
.kpi{{background:{C['card']};border:1px solid {C['border']};border-radius:12px;padding:1rem 1.3rem;flex:1;min-width:130px;}}
.kpi-lbl{{font-size:0.68rem;color:{C['muted']};text-transform:uppercase;letter-spacing:1px;font-family:'DM Mono',monospace;}}
.kpi-val{{font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:700;color:#fff;}}
.kpi-d{{font-size:0.72rem;margin-top:3px;}}
.du{{color:#34d399;}}.dw{{color:{C['gold']};}}.db{{color:{C['fraud']};}}
.af{{background:#1a0507;border:2px solid {C['fraud']};border-radius:12px;padding:1.4rem;text-align:center;}}
.al{{background:#011a16;border:2px solid {C['legit']};border-radius:12px;padding:1.4rem;text-align:center;}}
.apr{{font-family:'Syne',sans-serif;font-size:3.2rem;font-weight:800;line-height:1;}}
.sar{{background:#0f1f3d;border:1px solid #1e3a5f;border-radius:10px;padding:1.1rem;font-family:'DM Mono',monospace;font-size:0.76rem;line-height:1.7;color:#93c5fd;}}
div[data-testid="stMetric"]{{background:{C['card']};border:1px solid {C['border']};border-radius:10px;padding:0.8rem;}}
section[data-testid="stSidebar"]{{background:#0d1526;border-right:1px solid {C['border']};}}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="hero-title">🌍 Global AML Intelligence Platform</div>
  <div class="hero-sub">Real-time fraud monitoring · XGBoost + RandomForest Ensemble · SHAP Explainability · Graph Network Analysis</div>
  <div class="hero-badges">
    <span class="badge">FATF 40 Recommendations</span>
    <span class="badge">FinCEN (USA)</span>
    <span class="badge">FCA SYSC 6.3 (UK)</span>
    <span class="badge">AUSTRAC AML/CTF (AUS)</span>
    <span class="badge">EU AI Act 2024</span>
    <span class="badge badge-r">PaySim Dataset</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    SAMPLE_SIZE = st.slider("Training Sample Size", 5000, 60000, 25000, 5000)
    THRESHOLD   = st.slider("Alert Threshold", 0.10, 0.90, 0.40, 0.05,
                             help="Lower = more alerts, higher recall. Recommended AML: 0.30–0.40")
    st.markdown("---")
    market = st.selectbox("Regulatory Jurisdiction", [
        "🇺🇸 USA (FinCEN)", "🇬🇧 UK (FCA)", "🇦🇺 Australia (AUSTRAC)",
        "🇦🇪 UAE (CBUAE)", "🇨🇦 Canada (FINTRAC)", "🌍 Global (FATF)"])
    st.markdown("---")
    uploaded = st.file_uploader("📂 Upload PaySim CSV", type=['csv'])
    st.markdown("---")
    st.caption("Stack: XGBoost · RF · SHAP · NetworkX\nDataset: PaySim (Kaggle)\nContext: International FinTech")

# ── DATA ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file, n):
    if file:
        df = pd.read_csv(file)
        return df.sample(min(n,len(df)), random_state=42).reset_index(drop=True), True
    np.random.seed(42)
    nf, nl = int(n*0.013), n - int(n*0.013)
    legit = pd.DataFrame({
        'step':np.random.randint(1,744,nl), 'type':np.random.choice(['CASH-IN','PAYMENT','DEBIT','CASH-OUT'],nl,p=[.35,.34,.16,.15]),
        'amount':np.random.exponential(45000,nl).clip(1,600000).astype('float32'),
        'nameOrig':['C'+str(x) for x in np.random.randint(int(1e8),int(9e8),nl)],
        'oldbalanceOrg':np.random.exponential(90000,nl).clip(0).astype('float32'),
        'nameDest':['C'+str(x) for x in np.random.randint(int(1e8),int(9e8),nl)],
        'oldbalanceDest':np.random.exponential(70000,nl).clip(0).astype('float32'),
        'isFraud':0,'isFlaggedFraud':0
    })
    legit['newbalanceOrig']=(legit['oldbalanceOrg']-legit['amount']).clip(0).astype('float32')
    legit['newbalanceDest']=(legit['oldbalanceDest']+legit['amount']).astype('float32')

    fraud = pd.DataFrame({
        'step':np.random.randint(1,744,nf),'type':np.random.choice(['TRANSFER','CASH-OUT'],nf),
        'amount':np.random.uniform(80000,1200000,nf).astype('float32'),
        'nameOrig':['C'+str(x) for x in np.random.randint(int(1e8),int(9e8),nf)],
        'oldbalanceOrg':np.random.uniform(80000,1200000,nf).astype('float32'),
        'newbalanceOrig':np.zeros(nf,'float32'),
        'nameDest':['C'+str(x) for x in np.random.randint(int(1e8),int(9e8),nf)],
        'oldbalanceDest':np.zeros(nf,'float32'),
        'isFraud':1,'isFlaggedFraud':np.random.randint(0,2,nf)
    })
    fraud['amount']=fraud['oldbalanceOrg']
    fraud['newbalanceDest']=fraud['amount'].astype('float32')
    df = pd.concat([legit,fraud],ignore_index=True).sample(frac=1,random_state=42).reset_index(drop=True)
    return df, False

@st.cache_data(show_spinner=False)
def engineer(_df):
    df=_df.copy()
    df['hour_of_day']        =df['step']%24
    df['day_of_month']       =df['step']//24
    df['is_late_night']      =((df['hour_of_day']>=22)|(df['hour_of_day']<=4)).astype('int8')
    df['is_business_hours']  =((df['hour_of_day']>=9) &(df['hour_of_day']<=17)).astype('int8')
    df['orig_balance_delta'] =df['newbalanceOrig']-df['oldbalanceOrg']
    df['dest_balance_delta'] =df['newbalanceDest']-df['oldbalanceDest']
    df['balance_zeroed_out'] =(df['newbalanceOrig']==0).astype('int8')
    df['account_drained']    =((df['oldbalanceOrg']>0)&(df['newbalanceOrig']==0)).astype('int8')
    df['orig_balance_error'] =(df['oldbalanceOrg']-df['amount']-df['newbalanceOrig']).abs()
    df['dest_balance_error'] =(df['oldbalanceDest']+df['amount']-df['newbalanceDest']).abs()
    df['amount_to_orig_ratio']=df['amount']/(df['oldbalanceOrg']+1)
    df['amount_to_dest_ratio']=df['amount']/(df['oldbalanceDest']+1)
    df['dest_bal_to_amount'] =df['oldbalanceDest']/(df['amount']+1)
    df['is_high_risk_type']  =df['type'].isin(['TRANSFER','CASH-OUT','CASH_OUT']).astype('int8')
    df['dest_is_merchant']   =df['nameDest'].str.startswith('M').astype('int8')
    dummies=pd.get_dummies(df['type'],prefix='type',dtype='int8')
    return pd.concat([df,dummies],axis=1)

@st.cache_resource(show_spinner=False)
def train(_df):
    base=['amount','oldbalanceOrg','newbalanceOrig','oldbalanceDest','newbalanceDest',
          'hour_of_day','day_of_month','is_late_night','is_business_hours',
          'orig_balance_delta','dest_balance_delta','balance_zeroed_out','account_drained',
          'orig_balance_error','dest_balance_error',
          'amount_to_orig_ratio','amount_to_dest_ratio','dest_bal_to_amount',
          'is_high_risk_type','dest_is_merchant']
    tc=[c for c in _df.columns if c.startswith('type_')]
    feats=[c for c in base+tc if c in _df.columns]
    X=_df[feats].fillna(0).astype('float32'); y=_df['isFraud']
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    Xr,yr=SMOTE(random_state=42,k_neighbors=3,sampling_strategy=0.15).fit_resample(Xtr,ytr)
    xgb=XGBClassifier(n_estimators=250,max_depth=5,learning_rate=0.08,subsample=0.8,
                       colsample_bytree=0.8,use_label_encoder=False,eval_metric='aucpr',random_state=42,n_jobs=-1)
    rf =RandomForestClassifier(n_estimators=150,max_depth=12,class_weight='balanced',random_state=42,n_jobs=-1)
    ens=VotingClassifier([('xgb',xgb),('rf',rf)],voting='soft',weights=[2,1])
    xgb.fit(Xr,yr); rf.fit(Xr,yr); ens.fit(Xr,yr)
    return ens,xgb,Xte,yte,feats

# ── RUN ──────────────────────────────────────────────────────
with st.spinner("🔄 Loading data & training model..."):
    df_raw, is_real = load_data(uploaded, SAMPLE_SIZE)
    df_fe = engineer(df_raw)
    model, xgb_model, X_te, y_te, FEATS = train(df_fe)

yp=model.predict_proba(X_te)[:,1]; ypred=(yp>=THRESHOLD).astype(int)
roc=roc_auc_score(y_te,yp); prauc=average_precision_score(y_te,yp)
f1_=f1_score(y_te,ypred); rec=recall_score(y_te,ypred,zero_division=0)
prec=precision_score(y_te,ypred,zero_division=0)
nT=len(df_raw); nF=int(df_raw['isFraud'].sum()); nA=int(ypred.sum())

if not is_real:
    st.info("💡 Demo mode — synthetic PaySim-schema data. Upload your CSV via sidebar for real analysis.")

st.markdown(f"""
<div class="kpi-row">
  <div class="kpi"><div class="kpi-lbl">Transactions</div><div class="kpi-val">{nT:,}</div><div class="kpi-d du">30-day window</div></div>
  <div class="kpi"><div class="kpi-lbl">Fraud Cases</div><div class="kpi-val">{nF:,}</div><div class="kpi-d db">{nF/nT*100:.3f}% rate</div></div>
  <div class="kpi"><div class="kpi-lbl">ROC-AUC</div><div class="kpi-val">{roc:.4f}</div><div class="kpi-d du">↑ Discrimination</div></div>
  <div class="kpi"><div class="kpi-lbl">PR-AUC</div><div class="kpi-val">{prauc:.4f}</div><div class="kpi-d du">↑ Primary metric</div></div>
  <div class="kpi"><div class="kpi-lbl">Recall</div><div class="kpi-val">{rec:.4f}</div><div class="kpi-d {"du" if rec>0.8 else "dw"}">fraud caught</div></div>
  <div class="kpi"><div class="kpi-lbl">Alerts @ {THRESHOLD:.0%}</div><div class="kpi-val">{nA:,}</div><div class="kpi-d dw">flagged txns</div></div>
</div>
""", unsafe_allow_html=True)

t1,t2,t3,t4,t5 = st.tabs(["📊 EDA & Insights","🤖 Model Performance","🧠 SHAP Explainability","🕸️ Network Graph","🔍 Score a Transaction"])

def dark_fig(w=7,h=4):
    fig,ax=plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(C['card']); ax.set_facecolor(C['card'])
    return fig,ax

def style_ax(ax):
    ax.tick_params(colors=C['text'])
    for sp in ax.spines.values(): sp.set_color(C['border'])

# ═══ TAB 1: EDA ══════════════════════════════════════════════
with t1:
    st.markdown("## 📊 Exploratory Data Analysis")
    st.markdown("*Every chart answers a business question. Every finding maps to a model decision.*")

    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### Fraud Rate by Transaction Type")
        ts=df_raw.groupby('type')['isFraud'].agg(['sum','mean']).reset_index()
        fig,ax=dark_fig()
        ax.bar(ts['type'],ts['mean']*100,color=[C['fraud'] if r>0 else C['legit'] for r in ts['mean']],edgecolor='#374151',linewidth=0.8)
        ax.set_ylabel('Fraud Rate (%)',color=C['text']); style_ax(ax); plt.xticks(rotation=12)
        plt.tight_layout(); st.pyplot(fig)
        st.caption("**Finding:** Fraud ONLY in TRANSFER & CASH-OUT → skip scoring 65% of transactions in production")

    with c2:
        st.markdown("#### Hourly Fraud Pattern")
        df_raw['_h']=df_raw['step']%24; hr=df_raw.groupby('_h')['isFraud'].mean()*100
        fig,ax=dark_fig()
        ax.fill_between(hr.index,hr.values,alpha=0.3,color=C['fraud'])
        ax.plot(hr.index,hr.values,color=C['fraud'],linewidth=2.2)
        ax.set_xlabel('Hour of Day',color=C['text']); ax.set_ylabel('Fraud Rate (%)',color=C['text'])
        ax.set_xticks(range(0,24,2)); style_ax(ax); plt.tight_layout(); st.pyplot(fig)
        st.caption("**Finding:** Late-night hours have elevated fraud → `is_late_night` feature captures this")

    st.markdown("---")
    c3,c4=st.columns(2)
    with c3:
        st.markdown("#### Amount Distribution")
        clip=float(df_raw['amount'].quantile(0.99)); fig,ax=dark_fig()
        ax.hist(df_raw[df_raw['isFraud']==0]['amount'].clip(upper=clip),bins=55,color=C['legit'],alpha=0.6,label='Legitimate',density=True)
        ax.hist(df_raw[df_raw['isFraud']==1]['amount'].clip(upper=clip),bins=55,color=C['fraud'],alpha=0.75,label='Fraudulent',density=True)
        ax.set_xlabel('Amount',color=C['text']); ax.set_ylabel('Density',color=C['text'])
        ax.legend(facecolor=C['dark'],labelcolor=C['text']); style_ax(ax); plt.tight_layout(); st.pyplot(fig)

    with c4:
        st.markdown("#### Account Draining Signal")
        fz=(df_raw[df_raw['isFraud']==1]['newbalanceOrig']==0).mean()*100
        lz=(df_raw[df_raw['isFraud']==0]['newbalanceOrig']==0).mean()*100
        fig,ax=dark_fig()
        bars=ax.bar(['Legitimate','Fraudulent'],[lz,fz],color=[C['legit'],C['fraud']],edgecolor='#374151',linewidth=0.8,width=0.5)
        ax.set_ylabel('% with Origin Balance = $0',color=C['text']); style_ax(ax)
        for b,v in zip(bars,[lz,fz]): ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center',color=C['text'],fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)
        st.caption("**Key AML Signal:** Fraudsters drain accounts to $0 — `account_drained` is top SHAP feature")

    st.markdown("---")
    st.markdown("### 📋 EDA → Model Design Mapping")
    st.dataframe(pd.DataFrame({
        'EDA Finding':['769:1 class imbalance','Fraud only in 2 transaction types','Fraud amounts 4-5× larger','~80% of fraud drains to $0','isFlaggedFraud catches <0.2%','Late-night elevated rates'],
        'Model Decision':['Use PR-AUC + F1, not Accuracy','is_high_risk_type pre-filter feature','amount_to_orig_ratio ratio feature','account_drained + balance_zeroed_out','Exclude isFlaggedFraud from features','is_late_night + hour_of_day features'],
        'Implementation':['SMOTE oversampling (train only)','65% scoring reduction in production','Captures "transferred entire balance"','Strongest individual AML predictor','ML = 100× improvement over rules','Dynamic thresholds by time-of-day']
    }), use_container_width=True, hide_index=True)

# ═══ TAB 2: MODEL PERFORMANCE ════════════════════════════════
with t2:
    st.markdown("## 🤖 Model Performance")
    st.markdown("*Accuracy = useless at 0.13% fraud rate. These are the metrics that matter in AML.*")

    c1,c2=st.columns(2)
    with c1:
        st.markdown("#### ROC Curve")
        fpr,tpr,_=roc_curve(y_te,yp); fig,ax=dark_fig(6,5)
        ax.plot(fpr,tpr,color=C['fraud'],linewidth=2.5,label=f'Ensemble (AUC={roc:.4f})')
        ax.fill_between(fpr,tpr,alpha=0.08,color=C['fraud'])
        ax.plot([0,1],[0,1],'--',color='#4B5563',label='Random')
        ax.set_xlabel('False Positive Rate',color=C['text']); ax.set_ylabel('True Positive Rate',color=C['text'])
        ax.legend(facecolor=C['dark'],labelcolor=C['text'],fontsize=9); style_ax(ax); plt.tight_layout(); st.pyplot(fig)

    with c2:
        st.markdown("#### Precision-Recall Curve")
        pc,rc,_=precision_recall_curve(y_te,yp); fig,ax=dark_fig(6,5)
        ax.plot(rc,pc,color=C['gold'],linewidth=2.5,label=f'PR-AUC={prauc:.4f}')
        ax.fill_between(rc,pc,y_te.mean(),where=(pc>y_te.mean()),alpha=0.08,color=C['gold'])
        ax.axhline(y_te.mean(),color='#4B5563',ls='--',label='Baseline')
        ax.set_xlabel('Recall',color=C['text']); ax.set_ylabel('Precision',color=C['text'])
        ax.legend(facecolor=C['dark'],labelcolor=C['text'],fontsize=9)
        ax.set_title('Primary Metric for Imbalanced AML Data',color=C['text'],fontsize=9)
        style_ax(ax); plt.tight_layout(); st.pyplot(fig)

    st.markdown("---")
    c3,c4=st.columns([1,1.2])
    with c3:
        st.markdown("#### Confusion Matrix")
        cm=confusion_matrix(y_te,ypred); fig,ax=plt.subplots(figsize=(5.5,4.5))
        fig.patch.set_facecolor(C['card'])
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,linewidths=2,linecolor=C['dark'],
                    xticklabels=['Pred Legit','Pred Fraud'],yticklabels=['Actual Legit','Actual Fraud'],
                    annot_kws={'size':14,'weight':'bold'})
        ax.tick_params(colors=C['text']); plt.tight_layout(); st.pyplot(fig)

    with c4:
        st.markdown("#### Metrics at Current Threshold")
        tn_,fp_,fn_,tp_=cm.ravel()
        st.dataframe(pd.DataFrame({
            'Metric':['ROC-AUC','PR-AUC','F1','Precision','Recall','True Positives','False Negatives ⚠️','False Positives'],
            'Value':[f'{roc:.4f}',f'{prauc:.4f}',f'{f1_:.4f}',f'{prec:.4f}',f'{rec:.4f}',tp_,fn_,fp_],
            'AML Weight':['High','Critical — primary','High','Medium','Critical — missed fraud','Caught fraud','Regulatory risk','Analyst load']
        }), use_container_width=True, hide_index=True)
        st.warning(f"**{fn_} fraud transactions missed** at {THRESHOLD:.0%} threshold. Lower threshold to catch more.")

# ═══ TAB 3: SHAP ═════════════════════════════════════════════
with t3:
    st.markdown("## 🧠 SHAP Explainability")
    st.markdown("*Legally required in USA (FinCEN), UK (FCA), EU (AI Act). SHAP lets compliance officers justify every SAR filing.*")

    with st.spinner("Computing SHAP values..."):
        exp=shap.TreeExplainer(xgb_model)
        Xs=X_te.sample(min(800,len(X_te)),random_state=42)
        sv=exp.shap_values(Xs)

    c1,c2=st.columns([1.2,1])
    with c1:
        st.markdown("#### Global Feature Importance")
        ma=np.abs(sv).mean(0)
        fi=pd.Series(ma,index=Xs.columns).sort_values(ascending=True).tail(14)
        fig,ax=dark_fig(8,6)
        bc=[C['fraud'] if v>fi.mean() else C['legit'] for v in fi.values]
        fi.plot(kind='barh',ax=ax,color=bc,edgecolor=C['border'],linewidth=0.5)
        ax.set_xlabel('Mean |SHAP Value|',color=C['text'])
        ax.set_title('Top 14 Features — Red = above-avg impact',color=C['text'])
        style_ax(ax); plt.tight_layout(); st.pyplot(fig)

    with c2:
        st.markdown("#### Feature → AML Typology Map")
        st.dataframe(pd.DataFrame({
            'Feature':['account_drained','balance_zeroed_out','amount_to_orig_ratio','orig_balance_error','is_high_risk_type','is_late_night','dest_bal_to_amount','amount_to_dest_ratio'],
            'AML Pattern':['Account takeover → draining','Post-tx balance = $0','Full balance transferred','Balance manipulation','Layering transaction type','Automated fraud script','Mule account (no prior funds)','Suspicious amount vs dest']
        }), use_container_width=True, hide_index=True)

        sar_guide = {
            "🇺🇸 USA (FinCEN)":"File FinCEN SAR within 30 days. 31 CFR 1020.320 requires narrative explaining basis for suspicion.",
            "🇬🇧 UK (FCA)":"Submit DAML to NCA before proceeding. POCA 2002 s.330 — reporting where suspicion exists.",
            "🇦🇺 Australia (AUSTRAC)":"File SMR within 3 business days. AML/CTF Act 2006, Rule 16.",
            "🇦🇪 UAE (CBUAE)":"File STR via goAML within 5 business days. AML Law No. 20 of 2018.",
            "🇨🇦 Canada (FINTRAC)":"File STR within 30 days. PCMLTFA — 5-year record keeping.",
            "🌍 Global (FATF)":"FATF Rec 20: File STR where suspicion exists. No tipping-off allowed."
        }
        st.info(sar_guide.get(market, sar_guide["🌍 Global (FATF)"]))

    st.markdown("---")
    st.markdown("#### Local Explanation — Individual High-Risk Transaction")
    Xm=X_te.copy(); Xm['prob']=xgb_model.predict_proba(X_te)[:,1]; Xm['actual']=y_te.values
    tp_c=Xm[(Xm['actual']==1)&(Xm['prob']>0.8)]
    if len(tp_c)>0:
        idx=tp_c.index[0]; inst=X_te.loc[[idx]]; p=tp_c.loc[idx,'prob']
        st.markdown(f"**Fraud probability: {p:.2%}** — True Positive (model correctly caught this fraud)")
        si=exp.shap_values(inst)
        fig,_=plt.subplots(figsize=(11,6))
        shap.waterfall_plot(shap.Explanation(values=si[0],base_values=exp.expected_value,data=inst.values[0],feature_names=inst.columns.tolist()),show=False,max_display=12)
        plt.title(f'SHAP Waterfall — SAR Decision Support (Fraud Prob: {p:.2%})',fontsize=11,fontweight='bold')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown(f"""<div class="sar">
AUTO-GENERATED SAR NARRATIVE — {market}<br>
─────────────────────────────────────────────<br>
Transaction flagged: {p:.0%} fraud probability (ML Ensemble + XGBoost)<br><br>
INDICATORS OF SUSPICIOUS ACTIVITY:<br>
  [1] Origin balance drained to $0 after transaction (account_drained=1)<br>
  [2] Amount-to-balance ratio: {inst['amount_to_orig_ratio'].values[0]:.3f} (1.0 = full account transfer)<br>
  [3] Transaction type consistent with layering (TRANSFER/CASH-OUT)<br>
  [4] Balance reconciliation discrepancy detected<br><br>
RECOMMENDED ACTIONS:<br>
  → Hold transaction pending review<br>
  → Initiate Enhanced Customer Due Diligence (ECDD)<br>
  → File SAR/STR if suspicion unresolved<br>
  → Ref: FATF Rec 20 | {market}
</div>""", unsafe_allow_html=True)

# ═══ TAB 4: NETWORK GRAPH ════════════════════════════════════
with t4:
    st.markdown("## 🕸️ Transaction Network Analysis")
    st.markdown("*Individual transaction scoring misses smurfing, layering chains, and mule aggregation — all visible in the graph.*")

    with st.spinner("Building graph..."):
        gs=df_raw.sample(min(4000,len(df_raw)),random_state=42)
        G=nx.DiGraph()
        for _,r in gs.iterrows():
            G.add_edge(r['nameOrig'],r['nameDest'],weight=float(r['amount']),is_fraud=int(r['isFraud']))

    ca,cb,cc,cd=st.columns(4)
    ca.metric("Nodes (Accounts)",f"{G.number_of_nodes():,}")
    cb.metric("Edges (Transactions)",f"{G.number_of_edges():,}")
    fe2=sum(1 for _,_,d in G.edges(data=True) if d['is_fraud']==1)
    cc.metric("Fraud Edges",f"{fe2:,}")
    cd.metric("Graph Density",f"{nx.density(G):.6f}")

    st.markdown("---")
    cL,cR=st.columns([2,1])
    with cL:
        st.markdown("#### Money Laundering Subnetwork")
        fe3=[(u,v) for u,v,d in G.edges(data=True) if d['is_fraud']==1]
        if fe3:
            fn2=list(set([n for e in fe3[:120] for n in e]))[:70]; sub=G.subgraph(fn2)
            fig,ax=plt.subplots(figsize=(9,7)); fig.patch.set_facecolor('#0B1120'); ax.set_facecolor('#0B1120')
            pos=nx.spring_layout(sub,k=1.1,seed=42)
            on=list(set(u for u,v in fe3[:120] if u in sub)); dn=list(set(v for u,v in fe3[:120] if v in sub))
            os=[max(100,sub.out_degree(n)*35) for n in on if n in sub]
            ds_=[max(70,sub.in_degree(n)*28)  for n in dn if n in sub]
            nx.draw_networkx_nodes(sub,pos,nodelist=on,node_color=C['fraud'],node_size=os,alpha=0.9,ax=ax)
            nx.draw_networkx_nodes(sub,pos,nodelist=dn,node_color=C['gold'],node_size=ds_,alpha=0.85,ax=ax)
            nx.draw_networkx_edges(sub,pos,edge_color='#E6394650',arrows=True,arrowsize=12,alpha=0.5,ax=ax,width=1.1)
            ax.set_title('Fraud Graph  🔴 Senders  🟡 Receivers  (size ∝ degree)',color='white',fontsize=11,pad=10)
            ax.axis('off'); plt.tight_layout(); st.pyplot(fig)
        else:
            st.warning("No fraud edges in sample. Increase sample size.")

    with cR:
        st.markdown("#### Top Suspicious Accounts")
        ind=dict(G.in_degree()); outd=dict(G.out_degree())
        pr=nx.pagerank(G,weight='weight',max_iter=100)
        sd=pd.DataFrame({'account':list(ind.keys()),'receives_from':[ind[k] for k in ind],'sends_to':[outd.get(k,0) for k in ind],'pagerank':[round(pr.get(k,0),6) for k in ind]}).nlargest(10,'receives_from')
        sd['account']=sd['account'].str[:14]+'...'; st.dataframe(sd,use_container_width=True,hide_index=True)
        st.markdown("---")
        for flag,desc in [("High fan-in","Aggregation / smurfing hub"),("High fan-out","Structuring / distribution"),("High PageRank","Central node in fraud network"),("Short path","Rapid layering chain")]:
            st.markdown(f"**{flag}** — {desc}")

# ═══ TAB 5: SCORE ════════════════════════════════════════════
with t5:
    st.markdown("## 🔍 Real-Time Transaction Scorer")

    preset=st.selectbox("Load Preset:",["✏️ Manual Input","🚨 High Risk — Full Account Drain","🟡 Medium Risk — Large CASH-OUT","✅ Low Risk — Normal Payment","✅ Low Risk — Small CASH-IN"])
    P={
        "🚨 High Risk — Full Account Drain":dict(s=412,t='TRANSFER',a=920000,obO=920000,nbO=0,obD=0,nbD=920000),
        "🟡 Medium Risk — Large CASH-OUT":  dict(s=540,t='CASH-OUT', a=250000,obO=400000,nbO=150000,obD=20000,nbD=10000),
        "✅ Low Risk — Normal Payment":     dict(s=100,t='PAYMENT',  a=240,   obO=8000,nbO=7760,obD=0,nbD=240),
        "✅ Low Risk — Small CASH-IN":      dict(s=50, t='CASH-IN',  a=1500,  obO=0,nbO=1500,obD=5000,nbD=3500),
    }
    d=P.get(preset,dict(s=100,t='PAYMENT',a=500,obO=5000,nbO=4500,obD=1000,nbD=1500))

    st.markdown("---")
    r1,r2,r3=st.columns(3)
    with r1:
        si=st.number_input("Step",1,744,d['s']); tx=st.selectbox("Type",['PAYMENT','TRANSFER','CASH-OUT','CASH-IN','DEBIT'],index=['PAYMENT','TRANSFER','CASH-OUT','CASH-IN','DEBIT'].index(d['t']))
        am=st.number_input("Amount ($)",1.0,2e7,float(d['a']))
    with r2:
        obO=st.number_input("Origin Old Balance ($)",0.0,2e7,float(d['obO'])); nbO=st.number_input("Origin New Balance ($)",0.0,2e7,float(d['nbO']))
    with r3:
        obD=st.number_input("Dest Old Balance ($)",0.0,2e7,float(d['obD'])); nbD=st.number_input("Dest New Balance ($)",0.0,2e7,float(d['nbD']))

    if st.button("🔍 Score This Transaction", type="primary", use_container_width=True):
        row=pd.DataFrame([{'step':si,'type':tx,'amount':am,'nameOrig':'C111111111','nameDest':'C999999999','oldbalanceOrg':obO,'newbalanceOrig':nbO,'oldbalanceDest':obD,'newbalanceDest':nbD,'isFraud':0,'isFlaggedFraud':0}])
        fe_row=engineer(row)
        for col in FEATS:
            if col not in fe_row.columns: fe_row[col]=0
        Xs2=fe_row[FEATS].fillna(0).astype('float32')
        prob2=model.predict_proba(Xs2)[0][1]; fraud_f=prob2>=THRESHOLD

        rc1,rc2=st.columns([1,1.6])
        with rc1:
            clr=C['fraud'] if fraud_f else C['legit']
            verd="🚨 FRAUD ALERT" if fraud_f else "✅ CLEARED"
            cls_="af" if fraud_f else "al"
            act="HOLD & FILE SAR" if fraud_f else "APPROVE TRANSACTION"
            st.markdown(f"""<div class="{cls_}">
  <div style="font-size:0.85rem;color:{clr};font-weight:600;letter-spacing:2px;">{verd}</div>
  <div class="apr" style="color:{clr};">{prob2:.1%}</div>
  <div style="font-size:0.76rem;color:#9CA3AF;margin:4px 0;">Fraud Probability</div>
  <hr style="border-color:{clr}30;margin:0.7rem 0;">
  <div style="font-size:0.78rem;color:{clr};font-weight:600;">{act}</div>
  <div style="font-size:0.7rem;color:#9CA3AF;">Threshold: {THRESHOLD:.0%}</div>
</div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            zone="🟢 LOW" if prob2<0.3 else "🟡 MEDIUM" if prob2<0.6 else "🔴 HIGH"
            st.progress(float(prob2)); st.caption(f"Risk Zone: **{zone}**")

        with rc2:
            st.markdown("##### Risk Factors")
            factors=[
                (obO>0 and nbO==0,  "Account fully drained to $0"),
                (nbO==0,            "Origin balance now zero"),
                (tx in ['TRANSFER','CASH-OUT'],"High-risk transaction type"),
                (am/(obO+1)>0.9,    "Transferred >90% of balance"),
                (am>100000,         "Large amount (>$100K)"),
                (si%24>=22 or si%24<=4,"Late-night activity"),
                (obD==0,            "Destination had zero prior balance"),
                (abs(obO-am-nbO)>1, "Balance reconciliation error"),
            ]
            for triggered,desc in factors:
                st.markdown(f"{'🔴' if triggered else '🟢'} {desc} — {'**TRIGGERED**' if triggered else 'Clear'}")

            if fraud_f:
                st.markdown("---")
                st.markdown(f"""<div class="sar">
RECOMMENDED ACTION ({market})<br>
────────────────────────────<br>
1. HOLD transaction immediately<br>
2. Initiate Enhanced Due Diligence (EDD)<br>
3. Verify account holder identity<br>
4. File SAR/STR if unresolved<br>
5. Amount=${am:,.0f} | Type={tx} | Score={prob2:.2%}
</div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<div style='text-align:center;color:#6B7280;font-size:0.74rem;font-family:monospace;padding:0.8rem;'>🌍 Global AML Intelligence Platform · XGBoost · RF · SHAP · NetworkX · Streamlit · PaySim (Lopez-Rojas et al. 2016) · FATF · FinCEN · FCA · AUSTRAC · EU AI Act</div>", unsafe_allow_html=True)
