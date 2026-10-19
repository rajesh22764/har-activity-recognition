"""Step 2 - high-dimensionality study: all features vs PCA vs feature selection."""
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import REPORTS, SEED, load_features, savefig

sns.set_theme(style="whitegrid")
Xtr, ytr, str_ = load_features("train")
Xte, yte, _ = load_features("test")
cv = GroupKFold(n_splits=5)  # folds split by SUBJECT -> no person appears in train and val
lr = lambda: LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)

configs = {"All 561 features": make_pipeline(StandardScaler(), lr())}
for k in (10, 20, 35, 67, 155):
    configs[f"PCA {k}"] = make_pipeline(StandardScaler(), PCA(k, random_state=SEED), lr())
for k in (20, 50, 100, 200):
    configs[f"ANOVA top-{k}"] = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), lr())
configs["L1 (sparse) selection"] = make_pipeline(
    StandardScaler(),
    SelectFromModel(LinearSVC(penalty="l1", dual=False, C=0.05, max_iter=5000,
                                random_state=SEED)),
    lr())
configs["RF-importance top-60"] = make_pipeline(
    StandardScaler(),
    SelectFromModel(RandomForestClassifier(200, n_jobs=-1, random_state=SEED),
                    max_features=60, threshold=-float("inf")),
    lr())

rows = []
for name, pipe in configs.items():
    t = time.time()
    cvs = cross_val_score(pipe, Xtr, ytr, groups=str_, cv=cv, n_jobs=1)
    pipe.fit(Xtr, ytr)
    acc = accuracy_score(yte, pipe.predict(Xte))
    n_feat = Xtr.shape[1]
    if "pca" in pipe.named_steps:
        n_feat = pipe.named_steps["pca"].n_components_
    elif len(pipe.steps) == 3:
        n_feat = pipe.steps[1][1].transform(Xtr.iloc[:2]).shape[1]
    rows.append({"method": name, "n_features": n_feat, "cv_acc_mean": cvs.mean(),
                 "cv_acc_std": cvs.std(), "test_acc": acc, "seconds": time.time() - t})
    print(rows[-1])

res = pd.DataFrame(rows).round(4)
res.to_csv(REPORTS / "02_dimensionality.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(data=res, x="n_features", y="test_acc", hue="method", s=90, ax=ax)
ax.set(xscale="log", title="Accuracy vs. number of features (logistic regression)",
       xlabel="# features (log)", ylabel="test accuracy (held-out subjects)")
ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
savefig(fig, "05_dimensionality.png")
print(res.to_string())
