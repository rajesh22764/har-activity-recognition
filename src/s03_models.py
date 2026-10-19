"""Step 3 - classical ML comparison with subject-grouped cross-validation."""
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from common import MODELS, REPORTS, SEED, load_features, savefig

sns.set_theme(style="whitegrid")
Xtr, ytr, str_ = load_features("train")
Xte, yte, ste = load_features("test")
Xtr, Xte = Xtr.values, Xte.values  # arrays: LightGBM rejects special chars in column names
le = LabelEncoder().fit(ytr)
ytr_i, yte_i = le.transform(ytr), le.transform(yte)
cv = GroupKFold(n_splits=5)

models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
    "SVM (RBF)": make_pipeline(StandardScaler(), SVC(C=10, probability=True, random_state=SEED)),
    "Random Forest": RandomForestClassifier(300, n_jobs=-1, random_state=SEED),
    "LightGBM": lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                                   random_state=SEED, verbose=-1, n_jobs=8),
}
rows, preds, probas = [], {}, {}
for name, m in models.items():
    t = time.time()
    cvs = cross_val_score(m, Xtr, ytr_i, groups=str_, cv=cv)
    m.fit(Xtr, ytr_i)
    p = m.predict(Xte)
    preds[name], probas[name] = p, m.predict_proba(Xte)
    rows.append({"model": name, "cv_acc_mean": cvs.mean(), "cv_acc_std": cvs.std(),
                 "test_acc": accuracy_score(yte_i, p),
                 "test_macro_f1": f1_score(yte_i, p, average="macro"),
                 "seconds": time.time() - t})
    print(rows[-1], flush=True)
    joblib.dump(m, MODELS / f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib")

res = pd.DataFrame(rows).round(4).sort_values("test_acc", ascending=False)
res.to_csv(REPORTS / "03_models.csv", index=False)
print(res.to_string())

# leakage demonstration: same model, but random (row-level) CV mixes subjects
from sklearn.model_selection import StratifiedKFold
leak = cross_val_score(models["Random Forest"], Xtr, ytr_i,
                       cv=StratifiedKFold(5, shuffle=True, random_state=SEED)).mean()
grouped = res.set_index("model").loc["Random Forest", "cv_acc_mean"]
pd.Series({"random_row_cv": leak, "grouped_subject_cv": grouped}).to_csv(REPORTS / "03_leakage_demo.csv")
print(f"RF: random-row CV {leak:.4f} vs subject-grouped CV {grouped:.4f}")

best = res.iloc[0]["model"]
np.save(REPORTS / "pred_best_classical.npy", preds[best])
np.save(REPORTS / "proba_best_classical.npy", probas[best])
joblib.dump(le, MODELS / "label_encoder.joblib")
(REPORTS / "best_classical.txt").write_text(best)

fig, axs = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"wspace": 0.85})
sns.barplot(data=res, y="model", x="test_acc", ax=axs[0], color="#3b6ea5")
axs[0].set(xlim=(.85, 1), title="Test accuracy (unseen subjects)", ylabel="", xlabel="accuracy")
for i, v in enumerate(res.test_acc):
    axs[0].text(v + .001, i, f"{v:.3f}", va="center", fontsize=8)
cm = confusion_matrix(yte_i, preds[best], normalize="true")
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=axs[1], cbar=False,
            xticklabels=[c[:9] for c in le.classes_], yticklabels=le.classes_)
axs[1].set(title=f"{best} - normalized confusion matrix", xlabel="predicted", ylabel="true")
savefig(fig, "06_models.png")
