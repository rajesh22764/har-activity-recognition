# Wearable-Sensor Activity Monitoring — Findings Brief

**Audience:** a life-sciences client exploring smartphone/wearable sensors for remote patient monitoring
(e.g., mobility endpoints in trials, post-surgery recovery). **Data:** public UCI HAR dataset — 30 volunteers,
waist-worn phone, 6 activities, 10,299 two-and-a-half-second windows. Train and test contain **different people**.

## Bottom line
1. **A simple model is enough.** Logistic regression and an RBF-SVM both reach **≈95.4 %** accuracy on people the model
   never saw; deep learning on raw signals did *not* beat them (93.2 %, difference statistically significant, McNemar p<0.001).
2. **Laying is detected perfectly; the risk is in look-alike activities.** 96 % of errors fall in two groups —
   *sitting ↔ standing* (48 %) and *walking / upstairs / downstairs* (48 %).
3. **Reported accuracy depends heavily on how you validate.** The same random forest scores **98.2 %** with a random
   row split but **91.8 %** when whole people are held out. A random split leaks each person's signature into training and
   overstates real-world performance by ~6 points.
4. **Performance varies by person:** 89.6 % (subject 9) to 100 % (subject 24). Any deployment needs per-person monitoring.
5. **Cheap human-in-the-loop lift:** auto-accept only predictions with confidence ≥ 0.8 → 93 % of windows labelled at
   **97.8 %** accuracy; ≥ 0.9 → 89 % of windows at **98.5 %**. The rest go to review.

## Evidence

| Question | Method | Result |
|---|---|---|
| Is the data clean? | Missing-value / range / balance checks | 0 missing, all features in [-1, 1], classes 13–19 % each |
| Which features separate activities? | Kruskal-Wallis per feature, Benjamini-Hochberg FDR 1 % | 557 / 561 significant; strongest effect ε² ≈ 0.83 (`fBodyAccJerk-entropy()-X`) |
| Moving vs. still? | Mann-Whitney on `tBodyAcc-std()-X` | Near-complete separation (rank-biserial ≈ −1.00) |
| Sitting vs. standing? | Mann-Whitney + Cohen's d on all features | 333 significant, but only 15 features with \|d\| > 0.8 → weak, gravity-orientation-driven signal |
| Do we need all 561 features? | All vs. PCA vs. ANOVA / L1 / RF selection (subject-grouped CV) | All: 95.4 %. L1 (270 feats): 95.5 %. PCA-67: 92.6 %. PCA-10: 84.1 % → information is spread across many features |
| Best model? | 5-fold **GroupKFold by subject** + held-out test | SVM 95.4 % · LogReg 95.4 % · LightGBM 95.0 % · 1D-CNN (3-seed ens.) 93.2 % · Random Forest 92.8 % |
| What drives predictions? | SHAP (LightGBM) | Gravity-acceleration features ≈ 25 % of total importance (`tGravityAcc-min()-X` is #1) — i.e. *phone orientation*, which is why posture is easy and sit/stand is hard |
| Is accuracy equal across people? | Wilson CIs + χ² homogeneity test | Heterogeneous (χ² = 80, p ≈ 5e-14) — see caveat below |

## Caveats (please read)
- **30 volunteers, aged 19–48, one phone model, one waist position.** Results may not transfer to patients, older adults or other devices.
- Windows overlap by 50 % and come from the same people, so they are **not independent**; p-values (especially the
  per-subject χ² test) are overstated. Treat them as descriptive. Subject-grouped CV is the correct guard for the model metrics.
- Test set = only 9 people, so per-subject confidence intervals are wide (≈ ±4 points).
- The 1D-CNN was trained with light tuning only; a stronger architecture or more tuning could narrow the gap.
- The PCA component counts differ between the quick look (unscaled: 34/67/155 for 90/95/99 % variance) and the
  pipeline (standardized: 63/102/179). The pipeline numbers are the ones to use.
- AWS artifacts are prepared but **not deployed**. The LLM summary runs from a template unless `ANTHROPIC_API_KEY` is set (the template path is what was tested).

## Recommendation
Use a regularised linear model or SVM on engineered features as the baseline, validate by person, add a confidence-based
review queue, and collect data from the target population before making any clinical claim. Add orientation-independent
features if sitting/standing must be separated reliably.
