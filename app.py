from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(
    page_title="InsightForge AI",
    page_icon="IF",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_data.csv"
ACCENT = "#61a8ff"
PURPLE = "#a78bfa"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #f5f7fb; --muted: #98a4b7; --panel: rgba(19, 28, 45, .78); --line: rgba(148, 163, 184, .16); }
    .stApp { background: radial-gradient(circle at 12% 0%, #1e2b4f 0, #0a1020 34%, #070b14 100%); color: var(--ink); font-family: 'DM Sans', sans-serif; }
    [data-testid="stSidebar"] { background: rgba(8, 14, 28, .96); border-right: 1px solid var(--line); }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    h1 { font-size: 2.45rem !important; }
    .brand { padding: 1rem 0 1.6rem; }
    .brand-name { font: 700 1.5rem 'Space Grotesk', sans-serif; color: #fff; }
    .brand-mark { color: #79b7ff; }
    .brand-subtitle { color: var(--muted); font-size: .78rem; margin-top: .25rem; }
    .hero { padding: 1.3rem 1.5rem; margin-bottom: 1.2rem; border: 1px solid var(--line); border-radius: 16px; background: linear-gradient(110deg, rgba(33, 55, 95, .62), rgba(31, 23, 68, .48)); box-shadow: 0 18px 60px rgba(0,0,0,.18); }
    .hero p { color: #c2ccdc; max-width: 710px; margin-bottom: 0; }
    .metric-card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem; min-height: 106px; box-shadow: inset 0 1px rgba(255,255,255,.03); }
    .metric-label { color: var(--muted); font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }
    .metric-value { color: #fff; font: 700 1.75rem 'Space Grotesk', sans-serif; margin-top: .38rem; }
    .section-label { color: #8dbfff; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .stButton > button, .stDownloadButton > button { border-radius: 8px; border: 1px solid rgba(97,168,255,.45); background: rgba(54, 103, 177, .22); color: #eef6ff; }
    .stButton > button:hover, .stDownloadButton > button:hover { border-color: #8bc3ff; color: #fff; }
    div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stMetric"] { background: var(--panel); border: 1px solid var(--line); padding: .8rem; border-radius: 12px; }
    .small-note { color: var(--muted); font-size: .86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataset(uploaded_file: Any) -> tuple[pd.DataFrame | None, str]:
    """Load an uploaded CSV, or the bundled sample when no file is selected."""
    try:
        if uploaded_file is None:
            return pd.read_csv(SAMPLE_PATH), "Bundled sample_data.csv"
        return pd.read_csv(uploaded_file), uploaded_file.name
    except pd.errors.EmptyDataError:
        return None, "The selected CSV is empty."
    except Exception as exc:
        return None, f"Could not read the CSV: {exc}"


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(exclude=np.number).columns.tolist()


def metric_card(label: str, value: str | int) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = numeric_columns(features)
    categorical = categorical_columns(features)
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric))
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def infer_task(target: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(target) or target.nunique(dropna=True) <= 10:
        return "Classification"
    return "Regression"


def split_data(features: pd.DataFrame, target: pd.Series, task: str) -> tuple[Any, ...]:
    if len(features) < 8:
        raise ValueError("At least 8 rows are needed to create a useful train/test split.")
    if task == "Regression":
        clean_target = pd.to_numeric(target, errors="coerce")
        valid = clean_target.notna()
        features, clean_target = features.loc[valid], clean_target.loc[valid]
        if len(clean_target) < 8:
            raise ValueError("The regression target does not contain enough valid numeric values.")
        return train_test_split(features, clean_target, test_size=0.2, random_state=42)
    clean_target = target.astype("string").replace({"<NA>": np.nan}).dropna()
    features = features.loc[clean_target.index]
    if clean_target.nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")
    if clean_target.value_counts().min() < 2:
        return train_test_split(features, clean_target, test_size=0.2, random_state=42)
    return train_test_split(features, clean_target, test_size=0.2, random_state=42, stratify=clean_target)


def train_models(dataframe: pd.DataFrame, target_name: str, task: str) -> dict[str, Any]:
    features = dataframe.drop(columns=[target_name])
    target = dataframe[target_name]
    if features.shape[1] == 0:
        raise ValueError("Select a target while keeping at least one feature column.")
    if not numeric_columns(features) and not categorical_columns(features):
        raise ValueError("No usable feature columns were found.")
    x_train, x_test, y_train, y_test = split_data(features, target, task)
    if task == "Classification":
        candidates = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=150, random_state=42),
        }
    else:
        candidates = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=150, random_state=42),
        }
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(x_train)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted_models[model_name] = pipeline
        if task == "Classification":
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, predictions),
                    "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
                }
            )
        else:
            results.append(
                {
                    "Model": model_name,
                    "MAE": mean_absolute_error(y_test, predictions),
                    "MSE": mean_squared_error(y_test, predictions),
                    "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                    "R2": r2_score(y_test, predictions),
                }
            )
    return {
        "task": task,
        "target": target_name,
        "models": fitted_models,
        "results": pd.DataFrame(results),
        "x_test": x_test,
        "y_test": y_test,
        "features": features,
    }


def chart_figure(figsize: tuple[float, float] = (10, 4)) -> tuple[plt.Figure, plt.Axes]:
    figure, axis = plt.subplots(figsize=figsize)
    figure.patch.set_facecolor("#111a2a")
    axis.set_facecolor("#111a2a")
    axis.tick_params(colors="#b8c4d8")
    for spine in axis.spines.values():
        spine.set_color("#33435e")
    return figure, axis


def render_dashboard(dataframe: pd.DataFrame, source: str) -> None:
    st.markdown('<div class="section-label">Workspace overview</div>', unsafe_allow_html=True)
    st.title("InsightForge AI")
    st.markdown(
        '<div class="hero"><h3>Intelligent Machine Learning Analytics Platform</h3><p>Explore data quality, discover patterns, train transparent baseline models, and generate predictions from one focused workspace.</p></div>',
        unsafe_allow_html=True,
    )
    missing = int(dataframe.isna().sum().sum())
    numeric = numeric_columns(dataframe)
    categorical = categorical_columns(dataframe)
    cards = st.columns(5)
    values = [("Dataset", source), ("Rows", len(dataframe)), ("Columns", len(dataframe.columns)), ("Missing values", missing), ("Numerical / categorical", f"{len(numeric)} / {len(categorical)}")]
    for column, (label, value) in zip(cards, values):
        with column:
            metric_card(label, value)
    st.subheader("Dataset profile")
    if numeric:
        figure, axis = chart_figure()
        dataframe[numeric].mean().sort_values(ascending=False).plot(kind="bar", ax=axis, color=ACCENT)
        axis.set_ylabel("Mean value", color="#b8c4d8")
        axis.set_title("Average numerical values", color="white", loc="left")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    else:
        st.info("This dataset has no numerical columns for an overview chart.")


def render_analyzer(dataframe: pd.DataFrame, source: str) -> None:
    st.title("Dataset Analyzer")
    st.caption(f"Active source: {source}")
    if dataframe.empty or len(dataframe.columns) == 0:
        st.error("The dataset has no rows or columns. Upload a non-empty CSV to continue.")
        return
    cards = st.columns(4)
    for column, (label, value) in zip(cards, [("Rows", len(dataframe)), ("Columns", len(dataframe.columns)), ("Missing", int(dataframe.isna().sum().sum())), ("Duplicates", int(dataframe.duplicated().sum()))]):
        with column:
            metric_card(label, value)
    st.subheader("Preview")
    st.dataframe(dataframe, use_container_width=True, height=280)
    st.subheader("Column details")
    details = pd.DataFrame({"Column": dataframe.columns, "Data type": dataframe.dtypes.astype(str).values, "Missing": dataframe.isna().sum().values, "Unique": dataframe.nunique(dropna=True).values})
    st.dataframe(details, use_container_width=True, hide_index=True)
    with st.expander("Statistical summary"):
        st.dataframe(dataframe.describe(include="all").transpose(), use_container_width=True)
    st.download_button("Download processed dataset", dataframe.to_csv(index=False).encode("utf-8"), "insightforge_processed.csv", "text/csv")


def render_eda(dataframe: pd.DataFrame) -> None:
    st.title("Exploratory Data Analysis")
    numeric = numeric_columns(dataframe)
    if not numeric:
        st.warning("EDA charts require at least one numerical column.")
        return
    selected = st.selectbox("Histogram and box plot column", numeric)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    figure.patch.set_facecolor("#111a2a")
    for axis in axes:
        axis.set_facecolor("#111a2a")
        axis.tick_params(colors="#b8c4d8")
    sns.histplot(dataframe[selected].dropna(), kde=True, ax=axes[0], color=ACCENT)
    axes[0].set_title(f"Distribution of {selected}", color="white")
    sns.boxplot(x=dataframe[selected], ax=axes[1], color=PURPLE)
    axes[1].set_title(f"Spread of {selected}", color="white")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    if len(numeric) >= 2:
        st.subheader("Correlation and relationships")
        left, right = st.columns(2)
        with left:
            figure, axis = chart_figure((6, 4))
            sns.heatmap(dataframe[numeric].corr(), annot=True, fmt=".2f", cmap="mako", ax=axis, cbar=False)
            axis.set_title("Correlation heatmap", color="white")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
        with right:
            x_axis, y_axis = st.columns(2)
            x_column = x_axis.selectbox("X axis", numeric, key="eda_x")
            y_column = y_axis.selectbox("Y axis", [column for column in numeric if column != x_column] or numeric, key="eda_y")
            figure, axis = chart_figure((6, 4))
            sns.scatterplot(data=dataframe, x=x_column, y=y_column, ax=axis, color=ACCENT)
            axis.set_title(f"{x_column} vs {y_column}", color="white")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)


def render_ml(dataframe: pd.DataFrame) -> None:
    st.title("Machine Learning")
    if len(dataframe) < 8 or len(dataframe.columns) < 2:
        st.warning("Training needs at least 8 rows and one feature plus one target column.")
        return
    target = st.selectbox("Target column", dataframe.columns, key="ml_target")
    suggested = infer_task(dataframe[target])
    task = st.selectbox("Prediction type", ["Automatic", "Classification", "Regression"], index=0)
    resolved_task = suggested if task == "Automatic" else task
    st.caption(f"Selected task: {resolved_task}. Models are evaluated on a held-out test split with random_state=42.")
    if st.button("Train models", type="primary"):
        try:
            artifact = train_models(dataframe, target, resolved_task)
            st.session_state["model_artifact"] = artifact
            st.session_state.pop("prediction_result", None)
            st.success("Models trained successfully.")
        except Exception as exc:
            st.error(f"Training could not be completed: {exc}")
    artifact = st.session_state.get("model_artifact")
    if artifact and artifact["target"] == target and artifact["task"] == resolved_task:
        st.subheader("Measured model comparison")
        st.dataframe(artifact["results"].style.format({column: "{:.3f}" for column in artifact["results"].columns if column != "Model"}), use_container_width=True, hide_index=True)
        chosen = st.selectbox("Model for prediction", list(artifact["models"].keys()))
        st.session_state["selected_model"] = chosen
        if resolved_task == "Classification":
            model = artifact["models"][chosen]
            predictions = model.predict(artifact["x_test"])
            labels = model.classes_ if hasattr(model, "classes_") else np.unique(artifact["y_test"])
            figure, axis = chart_figure((6, 4))
            sns.heatmap(confusion_matrix(artifact["y_test"], predictions, labels=labels), annot=True, fmt="d", cmap="mako", ax=axis, cbar=False, xticklabels=labels, yticklabels=labels)
            axis.set_xlabel("Predicted", color="#b8c4d8")
            axis.set_ylabel("Actual", color="#b8c4d8")
            st.pyplot(figure)
            plt.close(figure)


def render_clustering(dataframe: pd.DataFrame) -> None:
    st.title("Clustering")
    numeric = numeric_columns(dataframe)
    if len(numeric) < 2:
        st.warning("K-Means visualization needs at least two numerical feature columns.")
        return
    selected = st.multiselect("Numerical features", numeric, default=numeric[: min(3, len(numeric))])
    clusters = st.slider("Number of clusters", min_value=2, max_value=min(8, max(2, len(dataframe) - 1)), value=3)
    if st.button("Train K-Means", type="primary"):
        if len(selected) < 2:
            st.error("Select at least two numerical features.")
            return
        if len(dataframe) < clusters:
            st.error("The dataset has fewer rows than the requested number of clusters.")
            return
        try:
            values = dataframe[selected].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < clusters:
                st.error("There are not enough complete rows in the selected features for clustering.")
                return
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values)
            estimator = KMeans(n_clusters=clusters, random_state=42, n_init=10)
            labels = estimator.fit_predict(scaled)
            clustered = dataframe.loc[values.index].copy()
            clustered["Cluster"] = labels
            st.session_state["clustered_data"] = clustered
            st.session_state["cluster_features"] = selected
            st.success("K-Means clusters created.")
        except Exception as exc:
            st.error(f"Clustering could not be completed: {exc}")
    clustered = st.session_state.get("clustered_data")
    if clustered is not None:
        st.subheader("Cluster profile")
        st.dataframe(clustered["Cluster"].value_counts().sort_index().rename("Rows"), use_container_width=True)
        features = st.session_state["cluster_features"]
        x_column, y_column = features[0], features[1]
        figure, axis = chart_figure()
        sns.scatterplot(data=clustered, x=x_column, y=y_column, hue="Cluster", palette="mako", ax=axis, s=70)
        axis.set_title("Cluster assignments", color="white")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        st.dataframe(clustered, use_container_width=True)
        st.download_button("Download clustered dataset", clustered.to_csv(index=False).encode("utf-8"), "insightforge_clustered.csv", "text/csv")


def render_prediction() -> None:
    st.title("Prediction")
    artifact = st.session_state.get("model_artifact")
    if not artifact:
        st.info("Train a model in Machine Learning before making a prediction.")
        return
    model_name = st.session_state.get("selected_model", next(iter(artifact["models"])))
    model = artifact["models"].get(model_name, next(iter(artifact["models"].values())))
    st.caption(f"Model: {model_name} | Target: {artifact['target']}")
    input_values: dict[str, Any] = {}
    for feature in artifact["features"].columns:
        series = artifact["features"][feature]
        if pd.api.types.is_numeric_dtype(series):
            median = float(pd.to_numeric(series, errors="coerce").median())
            input_values[feature] = st.number_input(feature, value=median if np.isfinite(median) else 0.0)
        else:
            options = series.dropna().astype(str).unique().tolist()
            input_values[feature] = st.selectbox(feature, options or ["Unknown"])
    if st.button("Predict", type="primary"):
        try:
            inputs = pd.DataFrame([input_values])
            prediction = model.predict(inputs)[0]
            result: dict[str, Any] = {"Prediction": prediction, "Model": model_name, "Target": artifact["target"]}
            st.session_state["prediction_result"] = result
            st.success(f"Prediction: {prediction}")
            if artifact["task"] == "Classification" and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(inputs)[0]
                classes = model.classes_
                probability_table = pd.DataFrame({"Class": classes, "Probability": probabilities}).sort_values("Probability", ascending=False)
                st.dataframe(probability_table.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")
    result = st.session_state.get("prediction_result")
    if result:
        st.download_button("Download prediction result", pd.DataFrame([result]).to_csv(index=False).encode("utf-8"), "insightforge_prediction.csv", "text/csv")


def render_about() -> None:
    st.title("About InsightForge AI")
    st.markdown("""
    InsightForge AI is a portfolio-ready analytics workspace for understanding tabular datasets and testing practical baseline machine-learning models.

    **Design principles**
    - Real calculations using scikit-learn pipelines, not simulated results.
    - Reproducible evaluation with a fixed random state.
    - Transparent preprocessing for numeric and categorical features.
    - Graceful validation so imperfect CSV files produce useful guidance.

    **Free deployment**
    The project runs locally with Streamlit and can be deployed on a free Python-compatible host such as Streamlit Community Cloud. GitHub stores the source code, but GitHub Pages cannot directly run a Python Streamlit application.
    """)


def main() -> None:
    if "active_dataset" not in st.session_state:
        sample_data, sample_source = load_dataset(None)
        st.session_state["active_dataset"] = sample_data
        st.session_state["dataset_source"] = sample_source

    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-name"><span class="brand-mark">IF</span> InsightForge AI</div><div class="brand-subtitle">Intelligent Machine Learning Analytics Platform</div></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
        if uploaded_file is not None:
            st.caption(f"Ready to upload: {uploaded_file.name}")
            if st.button("Upload dataset", type="primary", use_container_width=True):
                uploaded_data, uploaded_source = load_dataset(uploaded_file)
                if uploaded_data is None:
                    st.error(uploaded_source)
                elif uploaded_data.empty or len(uploaded_data.columns) == 0:
                    st.error("The selected CSV is empty. Upload a file containing headers and rows.")
                else:
                    st.session_state["active_dataset"] = uploaded_data
                    st.session_state["dataset_source"] = uploaded_source
                    st.session_state.pop("model_artifact", None)
                    st.session_state.pop("clustered_data", None)
                    st.session_state.pop("prediction_result", None)
                    st.rerun()
        st.divider()
        page = st.radio("Navigate", ["Dashboard", "Dataset Analyzer", "Exploratory Data Analysis", "Machine Learning", "Clustering", "Prediction", "About"], label_visibility="collapsed")

    dataframe = st.session_state.get("active_dataset")
    source = st.session_state.get("dataset_source", "Bundled sample_data.csv")
    if dataframe is None:
        st.error("The sample dataset could not be loaded. Upload a valid CSV to continue.")
        return
    if dataframe.empty or len(dataframe.columns) == 0:
        st.error("The active dataset is empty. Upload a CSV containing headers and rows.")
        return
    if page == "Dashboard":
        render_dashboard(dataframe, source)
    elif page == "Dataset Analyzer":
        render_analyzer(dataframe, source)
    elif page == "Exploratory Data Analysis":
        render_eda(dataframe)
    elif page == "Machine Learning":
        render_ml(dataframe)
    elif page == "Clustering":
        render_clustering(dataframe)
    elif page == "Prediction":
        render_prediction()
    else:
        render_about()


if __name__ == "__main__":
    main()
