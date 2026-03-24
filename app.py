import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Financial Fraud Detection System", page_icon="💳")

# ── Header ───────────────────────────────────────────────────
st.title("Financial Fraud Detection System")
st.markdown("**Built by:** Krupal Gohil &nbsp;|&nbsp; **Model:** Random Forest &nbsp;|&nbsp; **Dataset:** PaySim (Kaggle)")
st.markdown("---")

# ── About Project ────────────────────────────────────────────
with st.expander("About this Project"):
    st.markdown("""
    This project builds an end-to-end Financial fraud detection system
    on the PaySim synthetic financial transactions dataset containing 6.3 million transactions.

    **What it does:**
    - Detects fraudulent transactions using a Random Forest model trained on PaySim data
    - Explains why a transaction was flagged using feature-based risk analysis
    - Generates a Suspicious Activity Report (SAR) automatically for compliance teams

    **Tech Stack:** Python, Pandas, Seaborn, Scikit-learn, XGBoost, SHAP, Streamlit

    **Model Performance:**
    - ROC-AUC : 0.9994
    - Precision : 0.9891 — 98 out of every 100 alerts are genuine fraud
    - Recall : 0.9976 — catches almost all actual fraud cases
    - Overfitting check passed — train vs test F1 gap of only 0.0004

    **Business Impact:**
    In a real bank processing 1 million transactions daily, this system would automatically
    flag suspicious transactions and generate compliance reports — reducing manual review
    effort by up to 90% and cutting SAR writing time from hours to seconds.

    **Dataset:** PaySim — Lopez-Rojas et al. (2016). Synthetic Financial Datasets For Fraud Detection. Kaggle.
    """)

st.markdown("---")

# ── Load model ───────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('rf_model.pkl')

try:
    model = load_model()
    st.success("Model loaded successfully.")
except:
    st.error("rf_model.pkl not found. Run the notebook first and save the model.")
    st.stop()

# Features must match exactly what the model was trained on in the notebook
FEATURES = [
    'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest',
    'type_encoded', 'account_drained',
    'amount_to_balance', 'balance_change',
    'is_high_risk_type', 'hour_of_day', 'is_late_night'
]

# Same encoding used in the notebook LabelEncoder
TYPE_MAPPING = {
    'CASH-IN'  : 0,
    'CASH-OUT' : 1,
    'DEBIT'    : 2,
    'PAYMENT'  : 3,
    'TRANSFER' : 4
}

# ── SAR generator — same function as in the notebook ────────
def generate_sar(transaction, probability, reasons, report_id=1):
    """
    Generate a Suspicious Activity Report.
    Same function as defined in the notebook — kept identical
    so the report format is consistent everywhere.
    """
    report = f"""
SUSPICIOUS ACTIVITY REPORT
============================================================

  Date      : {pd.Timestamp.now().strftime('%d %B %Y')}
  Report ID : SAR-{pd.Timestamp.now().strftime('%Y%m%d')}-{report_id:04d}

  TRANSACTION DETAILS
  ------------------------------------------------------------
  Amount    : ${transaction['amount']:,.2f}
  Type      : {'TRANSFER / CASH-OUT' if transaction['is_high_risk_type']==1 else 'OTHER'}
  Time      : {'Late Night' if transaction['is_late_night']==1 else 'Normal Hours'}
  Balance   : ${transaction['oldbalanceOrg']:,.2f} -> ${transaction['newbalanceOrig']:,.2f}
  Account   : {'DRAINED TO ZERO' if transaction['account_drained']==1 else 'Not Drained'}

  FRAUD SCORE
  ------------------------------------------------------------
  Probability : {probability:.1%}
  Decision    : {'FLAGGED FOR REVIEW' if probability > 0.5 else 'CLEAR'}
  Model       : Random Forest (trained on PaySim)

  REASONS FLAGGED
  ------------------------------------------------------------
"""
    for i, r in enumerate(reasons, 1):
        report += f'  {i}. {r}\n'

    report += f"""
  RECOMMENDED ACTION
  ------------------------------------------------------------
  - Hold transaction immediately
  - Contact account holder for verification
  - File SAR with FIU-IND within 7 days
  - Reference : SAR-{pd.Timestamp.now().strftime('%Y%m%d')}-{report_id:04d}

============================================================
END OF REPORT
============================================================
"""
    return report


# ── Input form ───────────────────────────────────────────────
st.subheader("Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    amount       = st.number_input("Transaction Amount ($)",      0.0, 10000000.0, 1000.0)
    old_bal      = st.number_input("Sender Balance Before ($)",   0.0, 10000000.0, 5000.0)
    new_bal      = st.number_input("Sender Balance After ($)",    0.0, 10000000.0, 4000.0)

with col2:
    old_bal_dest = st.number_input("Receiver Balance Before ($)", 0.0, 10000000.0, 0.0)
    new_bal_dest = st.number_input("Receiver Balance After ($)",  0.0, 10000000.0, 1000.0)
    tx_type      = st.selectbox("Transaction Type",
                        ["PAYMENT", "TRANSFER", "CASH-OUT", "CASH-IN", "DEBIT"])
    hour         = st.slider("Hour of Transaction", 0, 23, 10)

check_btn = st.button("Check Transaction", type="primary", use_container_width=True)

# ── On button click ──────────────────────────────────────────
if check_btn:

    # Build features — exactly same logic as notebook
    is_high_risk = 1 if tx_type in ['TRANSFER', 'CASH-OUT'] else 0
    is_late      = 1 if hour >= 22 or hour <= 4 else 0
    drained      = 1 if old_bal > 0 and new_bal == 0 else 0
    amt_to_bal   = amount / (old_bal + 1)
    bal_change   = new_bal - old_bal
    type_enc     = TYPE_MAPPING.get(tx_type, 0)

    row = pd.DataFrame([{
        'amount'            : amount,
        'oldbalanceOrg'     : old_bal,
        'newbalanceOrig'    : new_bal,
        'oldbalanceDest'    : old_bal_dest,
        'newbalanceDest'    : new_bal_dest,
        'type_encoded'      : type_enc,
        'account_drained'   : drained,
        'amount_to_balance' : amt_to_bal,
        'balance_change'    : bal_change,
        'is_high_risk_type' : is_high_risk,
        'hour_of_day'       : hour,
        'is_late_night'     : is_late
    }])

    prob     = model.predict_proba(row[FEATURES])[0][1]
    is_fraud = prob >= 0.5

    st.markdown("---")

    # Result
    if is_fraud:
        st.error(f"SUSPICIOUS — Fraud Score: {prob:.1%}")
    else:
        st.success(f"LEGITIMATE — Fraud Score: {prob:.1%}")

    st.progress(float(prob))

    # Risk factors
    st.markdown("**Risk Factors:**")

    if drained:
        st.markdown("- Account was completely drained to zero")
    if amt_to_bal > 0.9:
        st.markdown("- More than 90% of balance was transferred")
    if is_high_risk:
        st.markdown("- TRANSFER and CASH-OUT are high risk types")
    if amount > 100000:
        st.markdown("- Transaction amount is unusually large")
    if is_late:
        st.markdown("- Late night transaction")
    if not any([drained, amt_to_bal > 0.9, is_high_risk, amount > 100000, is_late]):
        st.markdown("- No major risk factors found")

    # SAR report — only if flagged as fraud
    if is_fraud:
        st.markdown("---")
        st.subheader("SAR Report")

        reasons = []
        if drained         : reasons.append("Account was completely emptied to zero")
        if amt_to_bal > 0.9: reasons.append("Entire balance transferred at once")
        if is_high_risk    : reasons.append("High risk type — TRANSFER or CASH-OUT")
        if amount > 100000 : reasons.append("Unusually large transaction amount")
        if is_late         : reasons.append("Late night transaction")
        if not reasons     : reasons.append("Unusual pattern detected by ML model")

        transaction = {
            'amount'            : amount,
            'oldbalanceOrg'     : old_bal,
            'newbalanceOrig'    : new_bal,
            'is_high_risk_type' : is_high_risk,
            'is_late_night'     : is_late,
            'account_drained'   : drained,
        }

        sar = generate_sar(
            transaction = transaction,
            probability = prob,
            reasons     = reasons,
            report_id   = 1
        )

        st.text(sar)

        st.download_button(
            label     = "Download SAR Report",
            data      = sar,
            file_name = f"SAR_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime      = "text/plain"
        )

st.markdown("---")
st.caption("AML Fraud Detection System | Built by Krupal Gohil | Random Forest trained on PaySim Dataset | Python · Scikit-learn · Streamlit")