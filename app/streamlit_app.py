"""Streamlit demo: pick a held-out subject, see predicted activity timeline, confidence and summary.

Run:  streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import MODELS, REPORTS, load_features  # noqa: E402
from genai_summary import session_facts, write_summary  # noqa: E402

st.set_page_config(page_title="Wearable Activity Monitor", layout="wide")


@st.cache_resource
def load():
    model = joblib.load(MODELS / "svm_rbf.joblib")
    le = joblib.load(MODELS / "label_encoder.joblib")
    X, y, s = load_features("test")
    return model, le, X, y, s


model, le, X, y, subj = load()
st.title("Wearable-sensor activity monitor")
st.caption("UCI HAR dataset · SVM on 561 engineered features · predictions on **unseen** subjects")

c1, c2 = st.columns([1, 3])
with c1:
    sid = st.selectbox("Test subject", sorted(subj.unique()))
    thr = st.slider("Low-confidence threshold", 0.5, 0.95, 0.7, 0.05)
m = (subj == sid).values
proba = model.predict_proba(X.values[m])
pred = le.inverse_transform(proba.argmax(1))
conf = proba.max(1)
facts = session_facts(y[m], pred, conf, low_conf=thr)

with c1:
    st.metric("Windows", facts["n_windows"])
    st.metric("Accuracy vs. ground truth", f"{facts['accuracy_pct']}%")
    st.metric("Low-confidence windows", f"{facts['low_confidence_pct']}%")

with c2:
    classes = list(le.classes_)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = np.arange(len(pred))
    ax.step(t, [classes.index(v) for v in y[m]], where="mid", color="lightgrey", lw=5, label="truth")
    ax.step(t, [classes.index(v) for v in pred], where="mid", color="#3b6ea5", lw=1.5, label="predicted")
    ax.scatter(t[conf < thr], [classes.index(v) for v in pred[conf < thr]], c="#d1495b", s=14,
               zorder=3, label=f"confidence < {thr}")
    ax.set(yticks=range(6), yticklabels=classes, xlabel="window index (≈1.28 s each)")
    ax.legend(loc="upper right", fontsize=7, ncol=3)
    st.pyplot(fig)

st.subheader("Plain-English summary")
text, source = write_summary(facts)
st.write(text)
st.caption(f"Written by: {source}" + ("" if source == "claude" else "  (set ANTHROPIC_API_KEY to use Claude)"))

with st.expander("Model comparison and findings"):
    st.dataframe(pd.read_csv(REPORTS / "03_models.csv"), hide_index=True)
    st.image(str(ROOT / "reports" / "figures" / "07_shap.png"))
