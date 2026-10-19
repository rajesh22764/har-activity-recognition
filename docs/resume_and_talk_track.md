# Resume bullets and interview talk track

## Resume bullets (pick 3-4; every number comes from `reports/`)

**Project title:** Wearable-Sensor Activity Recognition for Remote Patient Monitoring | Python, scikit-learn, LightGBM, PyTorch, SHAP, Streamlit

- Built an end-to-end pipeline on 10,299 smartphone-sensor windows (561 features, 30 subjects); compared logistic regression, SVM, random forest, LightGBM and a 1D-CNN, reaching **95.4% accuracy on unseen subjects** with a simple SVM/linear model.
- Showed that random-row cross-validation **overstates accuracy by ~6 points** (98.2% vs. 91.8%) because of subject leakage; used subject-grouped CV throughout.
- Ran hypothesis tests (Kruskal-Wallis with FDR correction, Mann-Whitney, Cohen's d, McNemar) and a high-dimensionality study (PCA vs. ANOVA/L1/RF selection) to show information is spread across many features, not a few.
- Used SHAP and error analysis to trace 96% of errors to two look-alike groups (sitting/standing and the three walking types), driven by phone-orientation features; found per-subject accuracy of 89.6-100%.
- Designed a confidence-based review queue: auto-labelling windows at confidence of at least 0.8 covers 93% of data at 97.8% accuracy.
- Delivered a Streamlit dashboard with an LLM-written plain-English summary, plus SageMaker/Docker artifacts for deployment.

**One-line version:** Built and validated an activity-recognition model (95.4% on unseen subjects), quantified leakage from naive validation (+6 pts), and delivered an explainable Streamlit demo with a human-review workflow.

Do not claim: AWS deployment (artifacts only), results on patients (healthy volunteers only), or that the LLM summary ran in production (template path tested).

## 2-minute talk track

1. **Problem (15s):** "Pharma and healthcare teams want passive activity data from wearables, for example mobility endpoints in trials. I used a public dataset of 30 people wearing a phone to see how reliably a model can tell six activities apart."
2. **Approach (30s):** "I started with EDA and statistical tests to see which signals separate activities, then compared classical models against a deep network on the raw signals. I validated by holding out whole people, not random rows."
3. **Key finding (30s):** "A simple SVM or logistic regression hit about 95% and beat the CNN. Random splits would have shown 98%, which is misleading because the model learns each person's signature."
4. **Insight (30s):** "Errors are concentrated in sitting vs. standing and the walking types. SHAP shows the model leans on gravity/orientation features, which explains why: those postures look alike to a waist sensor."
5. **Impact (15s):** "I added a confidence threshold so uncertain windows go to human review, which lifts accuracy to about 98% on the rest, and a dashboard that summarizes a session in plain English."

## Likely questions and honest answers

- **Why did the CNN lose?** The engineered features already encode domain knowledge (gravity, frequency bands) and the dataset is small (21 training subjects). I tuned the CNN only lightly, so I would not claim deep learning is worse in general.
- **How do you know there is no leakage?** Windows from one person never appear in both train and validation; the official split is by subject and CV uses GroupKFold on subject.
- **Are the p-values trustworthy?** Windows overlap 50% and are autocorrelated, so p-values are overstated. I report effect sizes and treat the tests as descriptive.
- **Would this work on patients?** Not without new data. 30 healthy adults aged 19-48, one phone, one position.
- **How would you scale it?** Batch scoring through S3 plus Lambda or SageMaker Batch Transform, a SageMaker endpoint for real time. The handler is written but not deployed.
- **What would you do next?** Orientation-invariant features for sitting/standing, per-person calibration, more tuning of the sequence model, and data from the target population.
