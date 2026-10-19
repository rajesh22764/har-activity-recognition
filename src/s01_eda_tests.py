"""Step 1 - EDA and hypothesis testing."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from common import REPORTS, SEED, load_features, savefig

sns.set_theme(style="whitegrid")
Xtr, ytr, str_ = load_features("train")
Xte, yte, ste = load_features("test")
out = {}

# --- data quality -----------------------------------------------------------
out["shape_train"], out["shape_test"] = list(Xtr.shape), list(Xte.shape)
out["missing_values"] = int(Xtr.isna().sum().sum() + Xte.isna().sum().sum())
corr = np.corrcoef(Xtr.values, rowvar=False)
out["corr_pairs_gt_0.95"] = int((np.abs(corr[np.triu_indices(561, 1)]) > 0.95).sum())

# --- class balance ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.8))
bal = pd.concat([ytr.value_counts(normalize=True).rename("train"),
                 yte.value_counts(normalize=True).rename("test")], axis=1) * 100
bal.plot.barh(ax=ax, color=["#3b6ea5", "#e08a3c"])
ax.set(xlabel="% of windows", ylabel="", title="Class balance by split")
savefig(fig, "01_class_balance.png")

# --- PCA --------------------------------------------------------------------
Z = StandardScaler().fit_transform(Xtr)
pca = PCA(random_state=SEED).fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)
out["pca_components_for_90_95_99"] = [int((cum < t).sum() + 1) for t in (.9, .95, .99)]
comp = pca.transform(Z)[:, :2]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
sns.scatterplot(x=comp[:, 0], y=comp[:, 1], hue=ytr.values, s=7, ax=axs[0], linewidth=0)
axs[0].set(title="PCA projection (train)", xlabel="PC1", ylabel="PC2")
axs[0].legend(markerscale=2, fontsize=7)
axs[1].plot(range(1, 201), cum[:200], color="#3b6ea5")
for t in (.9, .95, .99):
    axs[1].axhline(t, ls="--", c="grey", lw=.7)
axs[1].set(title="Cumulative explained variance", xlabel="# components", ylabel="variance")
savefig(fig, "02_pca.png")

# --- hypothesis tests --------------------------------------------------------
# H1: for each feature, the distribution differs across the 6 activities (Kruskal-Wallis).
groups = [Xtr[ytr == a].values for a in sorted(ytr.unique())]
H, p = zip(*[stats.kruskal(*[g[:, j] for g in groups]) for j in range(561)])
H, p = np.array(H), np.array(p)
reject, p_adj, *_ = multipletests(p, method="fdr_bh", alpha=0.01)
eps2 = H / (len(Xtr) - 1)  # epsilon-squared effect size
kw = pd.DataFrame({"feature": Xtr.columns, "H": H, "p_adj_BH": p_adj, "epsilon_sq": eps2})
kw = kw.sort_values("epsilon_sq", ascending=False)
kw.to_csv(REPORTS / "kruskal_wallis_features.csv", index=False)
out["kw_features_significant_fdr1pct"] = int(reject.sum())
out["kw_top5"] = kw.head(5)[["feature", "epsilon_sq"]].round(3).to_dict("records")

# H2: moving vs static activities differ in body-acceleration variability.
col = "tBodyAcc-std()-X"
moving = Xtr.loc[ytr.str.startswith("WALKING"), col]
static = Xtr.loc[~ytr.str.startswith("WALKING"), col]
u = stats.mannwhitneyu(moving, static)
out["moving_vs_static"] = {"feature": col, "median_moving": float(moving.median()),
                           "median_static": float(static.median()), "p": float(u.pvalue),
                           "rank_biserial": float(1 - 2 * u.statistic / (len(moving) * len(static)))}

# H3: sitting vs standing - which features separate them at all? (the hard pair)
sit, stand = Xtr[ytr == "SITTING"], Xtr[ytr == "STANDING"]
pp = np.array([stats.mannwhitneyu(sit.iloc[:, j], stand.iloc[:, j]).pvalue for j in range(561)])
rej2, padj2, *_ = multipletests(pp, method="fdr_bh", alpha=0.01)
d = (sit.mean() - stand.mean()) / np.sqrt((sit.var() + stand.var()) / 2)
out["sit_vs_stand"] = {"significant_features": int(rej2.sum()),
                       "max_abs_cohens_d": float(d.abs().max()),
                       "features_with_abs_d_gt_0.8": int((d.abs() > .8).sum())}
walk = {a: Xtr[ytr == a] for a in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}
dd = {}
for a, b in [("WALKING", "WALKING_UPSTAIRS"), ("WALKING", "WALKING_DOWNSTAIRS"),
             ("WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS")]:
    di = (walk[a].mean() - walk[b].mean()) / np.sqrt((walk[a].var() + walk[b].var()) / 2)
    dd[f"{a} vs {b}"] = float(di.abs().max())
out["walking_pairs_max_abs_cohens_d"] = dd

# figure: key features
top = ["tGravityAcc-mean()-X", "tGravityAcc-min()-X", "tBodyAcc-std()-X", "angle(X,gravityMean)"]
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
for a, f in zip(axs, top):
    sns.boxplot(data=pd.DataFrame({"v": Xtr[f], "activity": ytr}), x="v", y="activity",
                ax=a, fliersize=1, linewidth=.8)
    a.set(title=f, xlabel="", ylabel="")
    if a is not axs[0]:
        a.set_yticklabels([])
savefig(fig, "03_key_features.png")

# figure: effect-size distribution
fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.hist(eps2, bins=40, color="#3b6ea5")
ax.set(title="Effect size of activity on each feature (Kruskal-Wallis epsilon²)",
       xlabel="epsilon²", ylabel="# features")
savefig(fig, "04_effect_sizes.png")

(REPORTS / "01_eda_results.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
