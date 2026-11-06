import csv
import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

from evaluation import MODEL_FILES, TARGET, evaluate_model, validate_data

ROOT = Path(__file__).resolve().parent
MAX_ROWS = 10000

st.set_page_config(page_title="Credit Default | Model Observatory", page_icon=":material/analytics:", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: #f7f9fa; color: #172622;}
    h1, h2, h3 {font-family: 'Trebuchet MS', sans-serif; letter-spacing: 0 !important;}
    h1 {font-size: 2rem !important;}
    [data-testid="stSidebar"] {background: #edf2ef;}
    [data-testid="stMetricValue"] {font-size: 1.65rem; white-space: normal; overflow-wrap: anywhere;}
    .stMainBlockContainer {padding-top: 2rem; max-width: 1440px;}
    @media (max-width: 700px) {
      h1 {font-size: 1.5rem !important;}
      .stMainBlockContainer {padding: 1.25rem 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models(signature):
    return {name: joblib.load(ROOT / "model" / filename) for name, filename in MODEL_FILES.items()}


@st.cache_data(show_spinner=False, max_entries=4)
def compare_models(frame, signature):
    features, target = validate_data(frame)
    results, predictions, probabilities = {}, {}, {}
    for name, model in load_models(signature).items():
        results[name], predictions[name], probabilities[name] = evaluate_model(model, features, target)
    comparison = pd.DataFrame.from_dict(results, orient="index")
    comparison.index.name = "ML Model Name"
    return comparison, predictions, probabilities


def read_csv(source):
    content = source.getvalue().decode("utf-8-sig")
    headers = next(csv.reader(io.StringIO(content)), [])
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate column names are not allowed.")
    return pd.read_csv(io.StringIO(content), nrows=MAX_ROWS + 1)


st.title("Credit Default")
st.caption("MODEL OBSERVATORY / UCI TAIWAN / 2005")

required_paths = [ROOT / "model" / filename for filename in MODEL_FILES.values()]
required_paths += [ROOT / "test_data.csv", ROOT / "model" / "results.json"]
if any(not path.exists() for path in required_paths):
    st.error("Trained artifacts are missing. Run `python train.py` from the project folder, then reload.")
    st.stop()

signature = tuple((path.name, path.stat().st_mtime_ns, path.stat().st_size) for path in required_paths)
metadata = json.loads((ROOT / "model" / "results.json").read_text())
with st.sidebar:
    st.header("Evaluation")
    source = st.radio("Test dataset", ["Bundled holdout", "Upload CSV"])
    uploaded_file = st.file_uploader("Labeled test CSV", type=["csv"])
    selected_model = st.selectbox("Model", list(MODEL_FILES))
    st.divider()
    st.download_button(
        "Test data", (ROOT / "test_data.csv").read_bytes(), "test_data.csv", "text/csv", icon=":material/download:"
    )
    st.caption(f"Training: {metadata['train_rows']:,} clients")
    st.caption(f"Holdout: {metadata['test_rows']:,} clients")
    st.caption("Positive class: default (1)")

try:
    if source == "Upload CSV":
        if uploaded_file is None:
            st.info("No test CSV selected.")
            st.stop()
        if uploaded_file.size > 10 * 1024 * 1024:
            raise ValueError("CSV files must be no larger than 10 MB.")
        frame = read_csv(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        frame = pd.read_csv(ROOT / "test_data.csv")
        dataset_name = "test_data.csv"
    if len(frame) > MAX_ROWS:
        raise ValueError(f"Upload at most {MAX_ROWS:,} test rows.")
    features, target = validate_data(frame)
except (ValueError, UnicodeError, pd.errors.ParserError, OSError) as error:
    st.error(str(error))
    st.stop()

with st.spinner("Evaluating saved models..."):
    try:
        comparison, predictions, probabilities = compare_models(frame, signature)
    except Exception as error:
        st.error(f"Saved-model evaluation failed: {error}")
        st.stop()

dataset_columns = st.columns(3)
dataset_columns[0].metric("Test clients", f"{len(frame):,}")
dataset_columns[1].metric("Observed default rate", f"{target.mean():.1%}")
dataset_columns[2].metric("Highest MCC", comparison["MCC"].idxmax())
st.caption(f"Dataset: {dataset_name}")

st.subheader("Model comparison")
st.dataframe(
    comparison.style.format("{:.4f}", na_rep="N/A").highlight_max(axis=0, color="#d9ece3"),
    width="stretch",
    height=250,
)
st.download_button(
    "Comparison", comparison.to_csv().encode(), "model_comparison.csv", "text/csv", icon=":material/download:"
)
if target.nunique() != 2:
    st.warning("AUC and ROC are undefined because this test data contains only one target class.")

st.subheader(selected_model)
metrics = comparison.loc[selected_model]
for group in [list(metrics.index[:3]), list(metrics.index[3:])]:
    for column, metric in zip(st.columns(3), group):
        value = metrics[metric]
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

diagnostics_tab, records_tab, experiment_tab = st.tabs(["Diagnostics", "Test records", "Experiment"])
with diagnostics_tab:
    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown("#### Confusion matrix")
        matrix = confusion_matrix(target, predictions[selected_model], labels=[0, 1])
        figure, axis = plt.subplots(figsize=(5, 3.8))
        axis.imshow(matrix, cmap="Greens")
        threshold = matrix.max() / 2
        for row_index in range(2):
            for column_index in range(2):
                axis.text(column_index, row_index, f"{matrix[row_index, column_index]:,}",
                          ha="center", va="center", fontsize=18,
                          color="white" if matrix[row_index, column_index] > threshold else "#172622")
        axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["No default", "Default"],
                 yticklabels=["No default", "Default"], xlabel="Predicted", ylabel="Actual")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with chart_columns[1]:
        st.markdown("#### ROC comparison")
        if target.nunique() == 2:
            figure, axis = plt.subplots(figsize=(5, 3.8))
            colors = ["#12664f", "#e16c52", "#387eaa", "#a88721", "#805b87"]
            for (name, scores), color in zip(probabilities.items(), colors):
                false_positive, true_positive, _ = roc_curve(target, scores)
                axis.plot(false_positive, true_positive, label=name, color=color,
                          linewidth=2.5 if name == selected_model else 1.2)
            axis.plot([0, 1], [0, 1], linestyle="--", color="#a4aaaa")
            axis.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02))
            axis.legend(loc="lower right", fontsize=7)
            figure.tight_layout()
            st.pyplot(figure, width="stretch")
            plt.close(figure)
        else:
            st.info("ROC unavailable for a single target class.")
    st.markdown("#### Classification report")
    report = classification_report(
        target, predictions[selected_model], labels=[0, 1],
        target_names=["No default (0)", "Default (1)"], output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"), width="stretch")

with records_tab:
    output = frame.copy()
    output["prediction"] = predictions[selected_model]
    output["default_probability"] = probabilities[selected_model]
    st.dataframe(output, width="stretch", hide_index=True)
    st.download_button("Predictions", output.to_csv(index=False).encode(), "predictions.csv",
                       "text/csv", icon=":material/download:")

with experiment_tab:
    st.json({key: value for key, value in metadata.items() if key != "metrics"})
    st.markdown("#### Original held-out results")
    st.dataframe(pd.DataFrame(metadata["metrics"]).T.style.format("{:.4f}"), width="stretch")

st.divider()
st.caption("Yeh, I. (2009) / UCI Default of Credit Card Clients / CC BY 4.0 / Educational benchmark")