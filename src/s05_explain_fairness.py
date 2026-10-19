"""Step 5 - explainability, error analysis, subject-level fairness, model comparison tests."""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
from statsmodels.stats.proportion import proportion_confint

from common import MODELS, REPORTS, SEED, load_features, savefig

sns.set_theme(style="whitegrid")
Xtr, ytr, _ = load_features("train")
Xte, yte, ste = load_features("test")
le = joblib.load(MODELS / "label_encoder.joblib")
y = le.transform(yte)
best = (REPORTS / "best_classical.txt").read_text()
p_svm = np.load(REPORTS / "pred_best_classical.npy")
pr_svm = np.load(REPORTS / "proba_best_classical.npy")
p_cnn = np.load(REPORTS / "pred_cnn.npy")
out = {"best_classical": best}

# --- 1. SHAP on LightGBM -------------------------------------------------------
lg = joblib.load(MODELS / "lightgbm.joblib")
samp = Xte.sample(600, random_state=SEED)
sv = shap.TreeExplainer(lg).shap_values(samp.values)
sv = np.stack(sv, -1) if isinstance(sv, list) else sv  # (n, f, k)
imp = pd.Series(np.abs(sv).mean((0, 2)), index=Xte.columns).sort_values(ascending=False)
imp.head(25).to_csv(REPORTS / "05_shap_top25.csv")
out["shap_top10"] = imp.head(10).round(4).to_dict()
grp = imp.groupby(lambda n: n.split("-")[0].split("(")[0]).sum().sort_values(ascending=False)
out["shap_share_by_signal_family_top6"] = (grp / grp.sum()).head(6).round(3).to_dict()

fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.3, 1]})
imp.head(15)[::-1].plot.barh(ax=axs[0], color="#3b6ea5")
axs[0].set(title="Top 15 features (mean |SHAP|, LightGBM)", xlabel="mean |SHAP|")
(grp / grp.sum()).head(8)[::-1].plot.barh(ax=axs[1], color="#e08a3c")
axs[1].set(title="Share of importance by signal family", xlabel="share")
savefig(fig, "07_shap.png")

# per-class driver: top feature per activity
per_cls = {}
for k, c in enumerate(le.classes_):
    s = pd.Series(np.abs(sv[:, :, k]).mean(0), index=Xte.columns).sort_values(ascending=False)
    per_cls[c] = s.head(3).index.tolist()
out["top3_features_per_activity"] = per_cls

# --- 2. error analysis -----------------------------------------------------------
cm = confusion_matrix(y, p_svm)
names = list(le.classes_)
errs = [(names[i], names[j], int(cm[i, j])) for i in range(6) for j in range(6) if i != j and cm[i, j]]
errs.sort(key=lambda t: -t[2])
out["top_confusions"] = errs[:6]
out["errors_total"] = int((p_svm != y).sum())
si, st = names.index("SITTING"), names.index("STANDING")
out["sit_stand_share_of_errors"] = round((cm[si, st] + cm[st, si]) / (p_svm != y).sum(), 3)
walk = [names.index(n) for n in ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS")]
out["walking_family_share_of_errors"] = round(
    sum(cm[i, j] for i in walk for j in walk if i != j) / (p_svm != y).sum(), 3)

# --- 3. subject-level fairness ---------------------------------------------------------
rows = []
for s in sorted(ste.unique()):
    m = (ste == s).values
    k, n = int((p_svm[m] == y[m]).sum()), int(m.sum())
    lo, hi = proportion_confint(k, n, method="wilson")
    rows.append({"subject": s, "n": n, "correct": k, "accuracy": k / n, "ci_lo": lo, "ci_hi": hi})
sub = pd.DataFrame(rows)
sub.to_csv(REPORTS / "05_subject_accuracy.csv", index=False)
table = np.array([[r.correct, r.n - r.correct] for r in sub.itertuples()])
chi2, pchi, *_ = stats.chi2_contingency(table)
out["subject_heterogeneity"] = {"chi2": float(chi2), "p": float(pchi),
                                "min_acc": float(sub.accuracy.min()),
                                "max_acc": float(sub.accuracy.max()),
                                "worst_subject": int(sub.loc[sub.accuracy.idxmin(), "subject"])}
fig, ax = plt.subplots(figsize=(8, 4))
ax.errorbar(range(len(sub)), sub.accuracy, yerr=[(sub.accuracy - sub.ci_lo).clip(lower=0), (sub.ci_hi - sub.accuracy).clip(lower=0)],
            fmt="o", color="#3b6ea5", capsize=4)
ax.axhline(accuracy_score(y, p_svm), ls="--", c="grey", label="overall")
ax.set(xticks=range(len(sub)), xticklabels=sub.subject, xlabel="test subject", ylabel="accuracy",
       title=f"Per-subject accuracy (95% Wilson CI) - {best}")
ax.legend()
savefig(fig, "08_subject_fairness.png")

# --- 4. SVM vs CNN: McNemar -----------------------------------------------------------------
a, b = (p_svm == y), (p_cnn == y)
n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
mc_p = stats.binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0
out["mcnemar_svm_vs_cnn"] = {"svm_right_cnn_wrong": n01, "cnn_right_svm_wrong": n10, "p": float(mc_p)}

# --- 5. confidence-based review (business angle) ------------------------------------------
conf = pr_svm.max(1)
rows = []
for t in (0, .5, .7, .8, .9, .95):
    keep = conf >= t
    rows.append({"threshold": t, "coverage": keep.mean(), "accuracy_on_kept": (p_svm[keep] == y[keep]).mean()})
sel = pd.DataFrame(rows).round(4)
sel.to_csv(REPORTS / "05_selective_prediction.csv", index=False)
out["selective_prediction"] = sel.to_dict("records")
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(sel.coverage * 100, sel.accuracy_on_kept * 100, "o-", color="#3b6ea5")
for r in sel.itertuples():
    ax.annotate(f"p≥{r.threshold}", (r.coverage * 100, r.accuracy_on_kept * 100), fontsize=7, xytext=(3, 4), textcoords="offset points")
ax.set(xlabel="% of windows auto-labelled", ylabel="accuracy on those (%)",
       title="Human-review trade-off: flag low-confidence windows")
savefig(fig, "09_selective_prediction.png")

(REPORTS / "05_explain_fairness.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(out, indent=2, default=str))
