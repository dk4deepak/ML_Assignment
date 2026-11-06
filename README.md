# Credit Default | Model Observatory

## a. Problem statement

Predict whether a credit card client will default on the next month's payment.
Compare five binary classifiers on the same held-out dataset and inspect their
performance in an interactive Streamlit application. This is an educational
benchmark, not a production lending decision system.

## b. Run locally

Use Python 3.12. Run these commands from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python -m unittest discover -s tests -t . -p 'test_*.py' -v
python -m streamlit run app.py
```

Open http://localhost:8501 after starting Streamlit. Training is a separate,
one-time step, never part of an app session. Once the artifacts are present,
only the final command is needed to launch the frontend.

Training uses `default of credit card clients.xls` in this folder when present,
otherwise it downloads the original workbook from UCI. To use a copy elsewhere:

```bash
python train.py --dataset '/path/to/default of credit card clients.xls'
```

On a managed macOS network, pip 24 may require
`python -m pip install --use-feature=truststore -r requirements.txt` to use the
system's trusted certificate authorities. Do not disable HTTPS verification.
If the workbook download fails certificate verification, download it through
your browser from the UCI link below and pass the extracted `.xls` file with
`--dataset`.

## c. Dataset description

- **Source:** [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).
- **Original file:** `default of credit card clients.xls`, read with `header=1`.
- **Size:** 30,000 Taiwanese credit card clients and 23 input features.
- **Target:** `default payment next month`; 1 means default, 0 means no default.
- **Features:** credit limit, sex, education, marital status, age, six repayment
  status fields, six bill amounts, and six previous-payment amounts.
- **Time period:** repayment histories from April through September 2005.
- **Missing values:** none in the original dataset; uploads must be complete.
- **Class balance:** 6,636 defaults (22.12%) and 23,364 non-defaults (77.88%).
- **Split:** 24,000 training rows and 6,000 held-out rows using stratification
  and `random_state=42`. The ID is excluded; all five models use identical rows.
- **Preprocessing:** original numeric codes are retained, including the dataset's
  additional education/marriage/status codes. StandardScaler is fitted only on
  training data for LR, KNN, Gaussian NB, and SVM. Each scaler is saved inside its
  model pipeline. The tree uses the unscaled numeric features.

The saved `test_data.csv` is the actual held-out experiment set, not synthetic
data. It contains all 23 feature columns and the target. An optional ID column
is ignored. Headers must match exactly; features are reordered before prediction.
Empty, nonnumeric, missing, infinite, or nonbinary-target data is rejected.

**Attribution:** Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI
Machine Learning Repository. https://doi.org/10.24432/C55S3H.
Licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The distributed CSV is a held-out subset with the ID removed; no other raw
feature transformations are applied to that CSV.

## d. Models used

| Model | Fixed configuration |
|---|---|
| Logistic Regression | StandardScaler, max_iter=2000 |
| Decision Tree | max_depth=6, min_samples_leaf=40 |
| KNN | StandardScaler, 25 neighbors, uniform weights |
| Naive Bayes | StandardScaler, GaussianNB with default smoothing |
| Support Vector Machine | StandardScaler, RBF SVC, C=1, gamma=scale, probability=True |

All random seeds are 42 where applicable. These are fixed baseline settings,
not settings selected by searching the test data. The SVM's probability
calibration uses training data only. The app uses each model's `predict` for
class labels and its positive-class `predict_proba` for AUC. In SVC these
labels need not exactly match a 0.5 calibrated-probability threshold.

### Comparison table

Scores are computed on the same 6,000-row holdout. Precision, recall and F1 use
default (1) as the positive class and binary averaging. AUC uses continuous
positive-class probabilities, not class labels. MCC summarizes both classes
and is the **predeclared winner criterion**, given class imbalance. Accuracy
alone should be compared with the 77.88% majority-class baseline.

<!-- METRICS:START -->
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8077 | 0.7076 | 0.6868 | 0.2396 | 0.3553 | 0.3244 |
| Decision Tree | 0.8168 | 0.7493 | 0.6681 | 0.3414 | 0.4519 | 0.3844 |
| KNN | 0.8078 | 0.7435 | 0.6261 | 0.3255 | 0.4284 | 0.3517 |
| Naive Bayes | 0.7525 | 0.7249 | 0.4515 | 0.5539 | 0.4975 | 0.3386 |
| Support Vector Machine | 0.8160 | 0.7210 | 0.6652 | 0.3384 | 0.4486 | 0.3809 |
<!-- METRICS:END -->

### Observations

<!-- OBSERVATIONS:START -->
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Scaled linear baseline; may miss nonlinear payment-history relationships. On this split: AUC 0.7076, default recall 0.2396, MCC 0.3244. |
| Decision Tree | Depth-limited nonlinear rules; leaf-size regularization reduces overfitting. On this split: AUC 0.7493, default recall 0.3414, MCC 0.3844. |
| KNN | Scaled distance-based classifier; neighborhood voting may favor the majority class. On this split: AUC 0.7435, default recall 0.3255, MCC 0.3517. |
| Naive Bayes | Gaussian conditional-independence assumptions are restrictive for correlated financial variables. On this split: AUC 0.7249, default recall 0.5539, MCC 0.3386. |
| Support Vector Machine | Scaled RBF kernel captures nonlinear relationships; fitting and inference are more expensive. On this split: AUC 0.7210, default recall 0.3384, MCC 0.3809. |
| Overall winner for this dataset | **Decision Tree**, using highest held-out MCC (0.3844) as the predeclared criterion. This is a split-specific comparison, not proof of statistical superiority. |
<!-- OBSERVATIONS:END -->

Single-class uploads have no defined AUC; the app displays N/A and still
calculates the other metrics. Zero-division precision/recall/F1 use 0. Confusion
matrices always show both labels. The app compares all saved models on either
the bundled holdout or an uploaded labeled test CSV without fitting anything.
For uploaded data, the highest-MCC label is descriptive of that upload only.

## Project files

```text
app.py                 Streamlit evaluation frontend
evaluation.py          Shared schema validation and metrics
train.py               Reproducible offline training and README generation
requirements.txt       Pinned runtime and training dependencies
README.md              Assignment report and instructions
tests/
  __init__.py          Test package for unittest discovery
  test_evaluation.py   Validation and metric tests
  test_app.py          Streamlit and saved-artifact integration tests
test_data.csv          Generated 6,000-row holdout
model/
  logistic_regression.joblib
  decision_tree.joblib
  knn.joblib
  naive_bayes.joblib
  support_vector_machine.joblib
  results.json         Measured metrics, split metadata and CSV SHA-256
```

The CSV and model directory are generated by training and must be committed
alongside the source before deployment. Saved pipelines include preprocessing;
do not upload or load untrusted pickle/joblib files. CSV is the only user upload
accepted by this app. The pinned scikit-learn version must match training.
The app has a 10,000-row evaluation limit to keep free-tier inference bounded.

## e. Streamlit Community Cloud deployment

1. Train locally and verify the models, `results.json`, measured README tables,
   and `test_data.csv` are present. Run the tests.
2. Push this project to your GitHub repository, including all five saved models
   and the test CSV. Do not publish unrelated enterprise automation files or secrets.
3. Visit https://streamlit.io/cloud and sign in with GitHub.
4. Choose **Create app / New app**, your repository, and the intended branch.
5. If this folder is the repository root, set the main file to `app.py`.
   If preserving this repository's layout, use `dk/ML/app.py` instead.
6. In advanced settings choose Python 3.12, then deploy. Streamlit installs the
   requirements located alongside the app.
7. Open the resulting public app URL and verify bundled data, CSV upload,
   model selection, six metrics, all-model comparison, confusion matrix,
   classification report, and downloads.

Publishing requires your GitHub and Streamlit accounts; no cloud deployment
is performed by the source code. Add the public URL to your submission after
deployment. This dataset includes sensitive demographic predictors and is
historical and geographically limited; fairness, calibration, external
validation, and lending-policy review are outside this classroom benchmark.