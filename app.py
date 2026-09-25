from __future__ import annotations

from html import escape
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
    page_title="DATA WATCH",
    page_icon="IF",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_data.csv"
ACCENT = "#61a8ff"
PURPLE = "#a78bfa"

TECHNOLOGY_TOOLTIPS = {
    "Python": {
        "code": "from pathlib import Path",
        "explanation": "Python is the application language used to connect the dashboard, data processing, and Machine Learning workflow.",
    },
    "Pandas": {
        "code": "import pandas as pd",
        "explanation": "Loads, cleans, manipulates, and analyzes CSV datasets.",
    },
    "NumPy": {
        "code": "import numpy as np",
        "explanation": "Provides numerical operations and array-based data processing.",
    },
    "Scikit-learn": {
        "code": "from sklearn.model_selection import train_test_split",
        "explanation": "Provides the project models, preprocessing, clustering, and evaluation metrics.",
    },
    "Matplotlib": {
        "code": "import matplotlib.pyplot as plt",
        "explanation": "Creates visualizations such as histograms, scatter plots, and regression graphs.",
    },
    "Seaborn": {
        "code": "import seaborn as sns",
        "explanation": "Creates statistical visualizations such as correlation heatmaps and analysis charts.",
    },
    "Streamlit": {
        "code": "import streamlit as st",
        "explanation": "Creates the interactive Python web dashboard.",
    },
    "K-Means": {
        "code": "from sklearn.cluster import KMeans",
        "explanation": "Groups similar data points into clusters.",
    },
    "StandardScaler": {
        "code": "from sklearn.preprocessing import StandardScaler",
        "explanation": "Normalizes numerical features before algorithms such as K-Means.",
    },
    "Linear Regression": {
        "code": "from sklearn.linear_model import LinearRegression",
        "explanation": "Predicts continuous values by learning relationships between inputs and outputs.",
    },
    "Random Forest": {
        "code": "from sklearn.ensemble import RandomForestClassifier",
        "explanation": "Combines multiple decision trees for classification or regression.",
    },
    "Train/Test Split": {
        "code": "from sklearn.model_selection import train_test_split",
        "explanation": "Divides data into training and testing sets for evaluation on unseen data.",
    },
    "Model Evaluation": {
        "code": "from sklearn.metrics import mean_squared_error, r2_score",
        "explanation": "Measures how well a trained Machine Learning model performs.",
    },
}

DATA_TYPE_TOOLTIPS = {
    "int64": ("64-bit integer data type.", "Whole-number numerical values such as age, income, and visit counts.", 'df["Age"].dtype', "Pandas"),
    "int32": ("32-bit integer data type.", "Whole-number numerical values stored with 32-bit precision.", 'df["Age"].dtype', "Pandas"),
    "float64": ("64-bit floating-point numerical data.", "Decimal values such as customer satisfaction scores.", 'df["CustomerSatisfaction"].dtype', "Pandas"),
    "float32": ("32-bit floating-point numerical data.", "Decimal values stored with 32-bit precision.", 'df["CustomerSatisfaction"].dtype', "Pandas"),
    "object": ("Usually text, string, or mixed data in a Pandas DataFrame.", "Categorical columns such as Membership or Churn.", 'df["Membership"].dtype', "Pandas"),
    "str": ("Text/string data stored by the current Pandas CSV reader.", "Categorical columns such as Membership or Churn.", 'df["Membership"].dtype', "Pandas"),
    "bool": ("Boolean data containing True or False values.", "Binary flags and yes/no style conditions.", 'df["is_active"].dtype', "Pandas"),
    "datetime64[ns]": ("Datetime values with nanosecond precision.", "Dates and times used for time-based analysis.", 'df["date"].dtype', "Pandas"),
    "category": ("A Pandas categorical data type.", "Repeated labels stored efficiently for categorical analysis.", 'df["Membership"].dtype', "Pandas"),
    "string": ("Pandas string data type.", "Text values stored explicitly as strings.", 'df["Membership"].dtype', "Pandas"),
}

DATASET_TOOLTIPS = {
    "Rows": ("Rows represent individual observations or records in the dataset.", "df.shape[0]", "Pandas"),
    "Columns": ("Columns represent variables or features in the dataset.", "df.shape[1]", "Pandas"),
    "Missing": ("The number of missing values in the dataset.", "df.isnull().sum().sum()", "Pandas"),
    "Missing values": ("The number of missing values in the dataset.", "df.isnull().sum().sum()", "Pandas"),
    "Duplicates": ("The number of duplicate records.", "df.duplicated().sum()", "Pandas"),
}

GRAPH_TOOLTIPS = {
    "Histogram": ("Shows the distribution of a numerical variable.", 'plt.hist(df["Age"])', "Matplotlib"),
    "Box Plot": ("Shows distribution, median, spread, and potential outliers.", 'sns.boxplot(x=df["Income"])', "Seaborn"),
    "Scatter Plot": ("Shows the relationship between two numerical variables.", 'plt.scatter(df["Age"], df["Income"])', "Matplotlib"),
    "Correlation Heatmap": ("Shows relationships between numerical variables.", 'corr = df.corr(numeric_only=True)\nsns.heatmap(corr, annot=True)', "Pandas + Seaborn"),
    "Regression Line": ("Visualizes the relationship between input and predicted output.", 'plt.scatter(X, y)\nplt.plot(X, model.predict(X))', "Matplotlib + Scikit-learn"),
    "Cluster Visualization": ("Displays observations grouped into clusters with similar feature patterns.", "clusters = model.fit_predict(X_scaled)", "Scikit-learn + Matplotlib"),
    "Average numerical values": ("Summarizes the mean of each numerical column for a quick dataset profile.", "df[numeric].mean().plot(kind=\"bar\")", "Pandas + Matplotlib"),
    "Confusion Matrix": ("Shows correct and incorrect classification predictions.", "cm = confusion_matrix(y_test, y_pred)", "Scikit-learn"),
}

METRIC_TOOLTIPS = {
    "Accuracy": ("The percentage of predictions that were correct.", "accuracy = accuracy_score(y_test, y_pred)", "Scikit-learn"),
    "Precision": ("How many predicted positive cases were actually positive.", "precision = precision_score(y_test, y_pred)", "Scikit-learn"),
    "Recall": ("How many actual positive cases were correctly detected.", "recall = recall_score(y_test, y_pred)", "Scikit-learn"),
    "F1 Score": ("A combined measure of precision and recall.", "f1 = f1_score(y_test, y_pred)", "Scikit-learn"),
    "MAE": ("The average absolute difference between actual and predicted values.", "mae = mean_absolute_error(y_test, y_pred)", "Scikit-learn"),
    "MSE": ("The average squared difference between actual and predicted values. Lower generally means smaller errors.", "mse = mean_squared_error(y_test, y_pred)", "Scikit-learn"),
    "RMSE": ("Prediction error in the same unit as the target.", "rmse = np.sqrt(mean_squared_error(y_test, y_pred))", "NumPy + Scikit-learn"),
    "R2": ("How much target variation is explained by the regression model.", "r2 = r2_score(y_test, y_pred)", "Scikit-learn"),
}

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
    .tech-stack { display: flex; flex-wrap: wrap; gap: .45rem; margin: .25rem 0 1.25rem; }
    .tech-item, .tooltip-label { position: relative; display: inline-block; cursor: help; }
    .tech-item { color: #c6d9ff; font-size: .72rem; letter-spacing: .08em; font-weight: 700; }
    .tech-item:not(:last-child)::after { content: '|'; color: #526783; margin-left: .45rem; }
    .tooltip-card { position: absolute; z-index: 1000; left: 0; top: calc(100% + .65rem); width: 350px; padding: .85rem 1rem; border: 1px solid rgba(119, 174, 255, .42); border-radius: 12px; background: rgba(10, 17, 32, .96); box-shadow: 0 14px 36px rgba(0,0,0,.38); backdrop-filter: blur(14px); opacity: 0; pointer-events: none; transform: translateY(-5px); transition: opacity .18s ease, transform .18s ease; }
    .tech-item:hover .tooltip-card, .tooltip-label:hover .tooltip-card, .workflow-step:hover .tooltip-card { opacity: 1; transform: translateY(0); }
    .tooltip-title { color: #fff; font-weight: 700; margin-bottom: .35rem; }
    .tooltip-purpose { color: #c2ccdc; font-size: .82rem; line-height: 1.4; margin-top: .45rem; }
    .tooltip-code { display: block; padding: .5rem .6rem; border-radius: 7px; background: #070d19; color: #9ed0ff; font: .76rem/1.45 Consolas, monospace; white-space: pre-wrap; }
    .info-label { position: relative; display: inline-block; cursor: help; color: #bcd5ff; font-weight: 700; }
    .info-label .tooltip-card { top: calc(100% + .5rem); width: 330px; }
    .details-table { width: 100%; border-collapse: collapse; font-size: .86rem; background: rgba(19,28,45,.62); }
    .details-table th, .details-table td { padding: .55rem .65rem; border-bottom: 1px solid var(--line); text-align: left; }
    .details-table th { color: #9fc5ff; font-size: .75rem; text-transform: uppercase; letter-spacing: .05em; }
    .details-table td { color: #dce6f5; }
    .nav-tech-note { display: none; }
    div[data-testid="stRadio"] label { position: relative; }
    div[data-testid="stRadio"] label:hover::after { position: absolute; left: 100%; top: 0; z-index: 1000; width: 260px; padding: .7rem .8rem; border: 1px solid rgba(119,174,255,.42); border-radius: 10px; background: rgba(10,17,32,.96); color: #d7e5ff; box-shadow: 0 14px 36px rgba(0,0,0,.38); white-space: pre-line; font-size: .75rem; pointer-events: none; }
    div[data-testid="stRadio"] label:nth-of-type(1):hover::after { content: "Dashboard\\A\\AStreamlit + Pandas\\Aimport streamlit as st\\Adataframe.shape\\A\\AInteractive dashboard and dataset metrics."; }
    div[data-testid="stRadio"] label:nth-of-type(2):hover::after { content: "Dataset Analyzer\\A\\APandas\\Apd.read_csv('data.csv')\\Adataframe.describe()\\A\\ACSV loading, types, missing values, duplicates, and summaries."; }
    div[data-testid="stRadio"] label:nth-of-type(3):hover::after { content: "Dataset Editor\\A\\APandas\\Adataframe.loc[row, column]\\A\\AEdit and manipulate DataFrame values."; }
    div[data-testid="stRadio"] label:nth-of-type(4):hover::after { content: "Exploratory Data Analysis\\A\\APandas + Matplotlib + Seaborn\\Asns.histplot(dataframe[column])\\Aplt.subplots()\\A\\ADistributions, correlations, and outliers."; }
    div[data-testid="stRadio"] label:nth-of-type(5):hover::after { content: "Machine Learning\\A\\AScikit-learn\\Atrain_test_split(X, y)\\Amodel.fit(X_train, y_train)\\A\\ATrain/test split, models, preprocessing, and evaluation."; }
    div[data-testid="stRadio"] label:nth-of-type(6):hover::after { content: "Clustering\\A\\AK-Means + StandardScaler\\Ascaled = StandardScaler().fit_transform(X)\\AKMeans(n_clusters=3).fit(scaled)\\A\\AScale numerical features and group similar records."; }
    div[data-testid="stRadio"] label:nth-of-type(7):hover::after { content: "Prediction\\A\\ATrained Scikit-learn model\\Amodel.predict(new_data)\\Amodel.predict_proba(new_data)\\A\\AUse feature inputs to generate a prediction."; }
    div[data-testid="stRadio"] label:nth-of-type(8):hover::after { content: "About\\A\\AComplete technology stack\\Aimport pandas as pd\\Afrom sklearn.model_selection import train_test_split\\A\\AProject workflow and learning references."; }
    .workflow { display: flex; flex-wrap: wrap; align-items: center; gap: .55rem; margin: 1rem 0; }
    .workflow-step { position: relative; padding: .65rem .8rem; border: 1px solid var(--line); border-radius: 9px; background: rgba(19,28,45,.78); color: #d7e5ff; cursor: help; }
    .workflow-arrow { color: #7794bb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def technology_tooltip(name: str, label: str | None = None) -> str:
    technology = TECHNOLOGY_TOOLTIPS[name]
    title = escape(label or name)
    code = escape(technology["code"])
    explanation = escape(technology["explanation"])
    return f'<span class="tech-item">{title}<span class="tooltip-card"><span class="tooltip-title">{title}</span><code class="tooltip-code">{code}</code><span class="tooltip-purpose">{explanation}</span></span></span>'


def info_tooltip(label: str, purpose: str, code: str, library: str, title: str | None = None) -> str:
    display_title = escape(title or label)
    return f'<span class="info-label">{escape(label)}<span class="tooltip-card"><span class="tooltip-title">{display_title}</span><span class="tooltip-purpose">{escape(purpose)}</span><code class="tooltip-code">{escape(code)}</code><span class="tooltip-purpose"><strong>Library:</strong> {escape(library)}</span></span></span>'


def dataset_tooltip(label: str) -> str:
    purpose, code, library = DATASET_TOOLTIPS[label]
    return info_tooltip(label, purpose, code, library)


def metric_tooltip(label: str) -> str:
    purpose, code, library = METRIC_TOOLTIPS.get(label, ("A measured model result.", "model.predict(X_test)", "Scikit-learn"))
    return info_tooltip(label, purpose, code, library)


def graph_tooltip(label: str, display_label: str | None = None) -> str:
    purpose, code, library = GRAPH_TOOLTIPS[label]
    return info_tooltip(display_label or label, purpose, code, library, title=label)


def render_graph_details(*graph_names: str) -> None:
    """Show real graph functions and code when the user clicks the code button."""
    for graph_name in graph_names:
        purpose, code, library = GRAPH_TOOLTIPS[graph_name]
        render_code_button(f"graph_{graph_name}", graph_name, code, library, purpose)


def render_code_button(code_id: str, title: str, code: str, library: str, purpose: str) -> None:
    """Display a compact button that reveals a real Python code example below it."""
    if st.button("View the code", key=f"view_code_{code_id}"):
        st.session_state["visible_code"] = code_id
    if st.session_state.get("visible_code") == code_id:
        code_lines = code.splitlines()
        context_lines = [
            "import pandas as pd",
            "import numpy as np",
            "from sklearn.model_selection import train_test_split",
            "df = pd.read_csv(\"data.csv\")",
            "features = df.drop(columns=[\"target\"])",
            "target = df[\"target\"]",
            "X_train, X_test, y_train, y_test = train_test_split(",
            "    features, target, test_size=0.2, random_state=42",
            ")",
        ]
        if len(code_lines) < 10:
            code_lines = context_lines[: 10 - len(code_lines)] + code_lines
        code_lines = code_lines[:15]
        st.markdown(f"**{title}**  \n**Library:** {library}  \n**Purpose:** {purpose}")
        st.code("\n".join(code_lines), language="python")


def dtype_tooltip(dtype: Any) -> str:
    dtype_name = str(dtype)
    purpose, used_for, code, library = DATA_TYPE_TOOLTIPS.get(
        dtype_name,
        (f"Pandas data type {dtype_name}.", "Values stored in this dataset column.", "df[column].dtype", "Pandas"),
    )
    return info_tooltip(dtype_name, f"Meaning: {purpose}\n\nUsed for: {used_for}", code, library, title=dtype_name.upper())


def column_tooltip(column: str, dtype: Any) -> str:
    purpose_map = {
        "Age": "Represents customer age.",
        "Income": "Customer income used as a numerical ML feature.",
        "SpendingScore": "Represents customer spending behavior.",
        "PurchaseFrequency": "Represents how often the customer purchases.",
        "WebsiteVisits": "Represents customer website activity.",
        "CustomerSatisfaction": "Represents the customer's satisfaction rating.",
        "Churn": "Represents whether the customer left the service.",
        "Membership": "Represents the customer's membership category.",
    }
    purpose = purpose_map.get(column, "A dataset variable used for analysis and, when suitable, Machine Learning features.")
    return info_tooltip(column, f"Data type: {dtype}\n\nPurpose: {purpose}", f'df[{column!r}]', "Pandas", title=column.upper())


def render_technology_stack() -> None:
    items = " ".join(technology_tooltip(name) for name in ["Python"] if name in TECHNOLOGY_TOOLTIPS)
    items += " ".join(technology_tooltip(name) for name in ["Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "Streamlit"])
    st.markdown(f'<div class="tech-stack">{items}</div>', unsafe_allow_html=True)


def render_section_title(title: str, technologies: list[str], purpose: str) -> None:
    cards = " ".join(technology_tooltip(name) for name in technologies)
    st.markdown(f'<div class="tooltip-label"><h1>{escape(title)}</h1><span class="tooltip-card"><span class="tooltip-title">{escape(title)}</span><span class="tooltip-purpose"><strong>Libraries:</strong> {cards}<br><br>{escape(purpose)}</span></span></div>', unsafe_allow_html=True)
    code = "\n\n".join(TECHNOLOGY_TOOLTIPS[name]["code"] for name in technologies if name in TECHNOLOGY_TOOLTIPS)
    render_code_button(f"section_{title}", title, code, " + ".join(technologies), purpose)


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


def clear_dataset_artifacts() -> None:
    """Clear results that were trained from an older dataset version."""
    for key in ("model_artifact", "clustered_data", "cluster_features", "prediction_result", "selected_model"):
        st.session_state.pop(key, None)


def prepare_edited_dataset(edited_data: pd.DataFrame, original_data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate editor output while preserving the original column order and types where possible."""
    edited = edited_data.reindex(columns=original_data.columns).copy()
    warnings: list[str] = []
    empty_rows = edited.isna().all(axis=1)
    if empty_rows.any():
        edited = edited.loc[~empty_rows].copy()
        warnings.append(f"Removed {int(empty_rows.sum())} completely empty row(s).")

    for column in original_data.columns:
        original_series = original_data[column]
        if pd.api.types.is_numeric_dtype(original_series):
            converted = pd.to_numeric(edited[column], errors="coerce")
            invalid_values = edited[column].notna() & converted.isna()
            if invalid_values.any():
                warnings.append(f"Column '{column}' contains non-numeric edits; those values were treated as missing for ML.")
            edited[column] = converted
            if pd.api.types.is_integer_dtype(original_series) and not edited[column].isna().any():
                edited[column] = edited[column].astype(original_series.dtype)
        elif pd.api.types.is_string_dtype(original_series) and not pd.api.types.is_object_dtype(original_series):
            edited[column] = edited[column].astype("string")

    if edited.empty:
        raise ValueError("The edited dataset must contain at least one non-empty row.")
    return edited, warnings


def numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(exclude=np.number).columns.tolist()


def metric_card(label: str, value: str | int, tooltip: str | None = None) -> None:
    label_html = tooltip or escape(label)
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label_html}</div><div class="metric-value">{escape(str(value))}</div></div>',
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
    render_section_title("DATA WATCH", ["Streamlit", "Pandas"], "Streamlit creates the dashboard and Pandas handles dataset information.")
    render_technology_stack()
    st.markdown(
        '<div class="hero"><h3>Using Machine Learning</h3><p>Explore data quality, discover patterns, train transparent baseline models, and generate predictions from one focused workspace.</p></div>',
        unsafe_allow_html=True,
    )
    missing = int(dataframe.isna().sum().sum())
    numeric = numeric_columns(dataframe)
    categorical = categorical_columns(dataframe)
    cards = st.columns(5)
    values = [("Dataset", source), ("Rows", len(dataframe)), ("Columns", len(dataframe.columns)), ("Missing values", missing), ("Numerical / categorical", f"{len(numeric)} / {len(categorical)}")]
    for column, (label, value) in zip(cards, values):
        with column:
            metric_card(label, value, dataset_tooltip(label) if label in DATASET_TOOLTIPS else None)
    st.markdown(graph_tooltip("Average numerical values", "Dataset profile"), unsafe_allow_html=True)
    if numeric:
        figure, axis = chart_figure()
        dataframe[numeric].mean().sort_values(ascending=False).plot(kind="bar", ax=axis, color=ACCENT)
        axis.set_ylabel("Mean value", color="#b8c4d8")
        axis.set_title("Average numerical values", color="white", loc="left")
        axis.tick_params(axis="x", rotation=25)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
        render_graph_details("Average numerical values")
    else:
        st.info("This dataset has no numerical columns for an overview chart.")


def render_analyzer(dataframe: pd.DataFrame, source: str) -> None:
    render_section_title("Dataset Analyzer", ["Pandas"], "Pandas loads CSV data, inspects data types, finds missing values and duplicates, and creates statistical summaries.")
    st.caption(f"Active source: {source}")
    if dataframe.empty or len(dataframe.columns) == 0:
        st.error("The dataset has no rows or columns. Upload a non-empty CSV to continue.")
        return
    cards = st.columns(4)
    for column, (label, value) in zip(cards, [("Rows", len(dataframe)), ("Columns", len(dataframe.columns)), ("Missing", int(dataframe.isna().sum().sum())), ("Duplicates", int(dataframe.duplicated().sum()))]):
        with column:
            metric_card(label, value, dataset_tooltip(label))
    st.subheader("Preview")
    st.dataframe(dataframe, use_container_width=True, height=280)
    st.subheader("Column details")
    details_rows = []
    for column in dataframe.columns:
        dtype = dataframe[column].dtype
        details_rows.append(
            f"<tr><td>{column_tooltip(str(column), dtype)}</td><td>{dtype_tooltip(dtype)}</td><td>{int(dataframe[column].isna().sum())}</td><td>{int(dataframe[column].nunique(dropna=True))}</td></tr>"
        )
    details_html = "".join(details_rows)
    st.markdown(f'<table class="details-table"><thead><tr><th>Column</th><th>Data type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>{details_html}</tbody></table>', unsafe_allow_html=True)
    st.caption("Hover over a column name or data type for a Pandas explanation.")
    with st.expander("Statistical summary"):
        st.markdown(info_tooltip("Statistics", "Pandas statistical analysis generates descriptive statistics for numerical columns.", "df.describe()", "Pandas"), unsafe_allow_html=True)
        render_code_button("statistics", "Statistics", "summary = df.describe()", "Pandas", "Generates descriptive statistics for numerical columns.")
        st.dataframe(dataframe.describe(include="all").transpose(), use_container_width=True)
    render_code_button("processed_download", "Download processed dataset", "csv = df.to_csv(index=False)\nst.download_button(\"Download\", csv)", "Pandas + Streamlit", "Saves the processed DataFrame as a CSV file.")
    st.download_button("Download processed dataset", dataframe.to_csv(index=False).encode("utf-8"), "insightforge_processed.csv", "text/csv", help="Pandas + Streamlit: save the processed DataFrame as CSV.")


def render_editor(dataframe: pd.DataFrame) -> None:
    render_section_title("Dataset Editor", ["Pandas"], "Pandas DataFrames hold the editable dataset values and preserve the active data used throughout the application.")
    st.caption("Edit values, add rows, or delete rows without changing the column names.")
    original_data = st.session_state.get("original_dataset", dataframe).copy()
    notice = st.session_state.pop("editor_notice", None)
    if notice:
        st.success(notice)

    editor_key = f"dataset_editor_{st.session_state.get('editor_version', 0)}"
    edited_data = st.data_editor(
        dataframe,
        num_rows="dynamic",
        use_container_width=True,
        key=editor_key,
    )
    render_code_button("data_editor", "Dataset Editor", "edited_df = st.data_editor(df, num_rows=\"dynamic\")", "Streamlit + Pandas", "Edits DataFrame cells and supports adding or deleting rows.")
    apply_column, reset_column = st.columns(2)
    with apply_column:
        apply_changes = st.button("Apply Changes", type="primary", use_container_width=True, help="Validate and save the edited Pandas DataFrame as the active dataset.")
    with reset_column:
        reset_dataset = st.button("Reset Dataset", use_container_width=True, help="Discard unsaved edits and restore the originally uploaded CSV.")

    if apply_changes:
        try:
            prepared_data, warnings = prepare_edited_dataset(edited_data, original_data)
            st.session_state["active_dataset"] = prepared_data
            clear_dataset_artifacts()
            st.session_state["editor_notice"] = "Dataset changes applied successfully."
            for warning in warnings:
                st.session_state.setdefault("editor_warnings", []).append(warning)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    if reset_dataset:
        st.session_state["active_dataset"] = original_data.copy()
        st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
        st.session_state.pop("editor_warnings", None)
        clear_dataset_artifacts()
        st.session_state["editor_notice"] = "Dataset reset to the originally uploaded CSV."
        st.rerun()

    for warning in st.session_state.pop("editor_warnings", []):
        st.warning(warning)
    st.download_button(
        "Download Edited CSV",
        dataframe.to_csv(index=False).encode("utf-8"),
        "insightforge_edited_data.csv",
        "text/csv",
        help="Pandas + Streamlit: save the current edited DataFrame as a CSV file.",
        use_container_width=True,
    )


def render_eda(dataframe: pd.DataFrame) -> None:
    render_section_title("Exploratory Data Analysis", ["Pandas", "Matplotlib", "Seaborn"], "These libraries create distributions, correlations, scatter plots, and outlier views for data analysis.")
    numeric = numeric_columns(dataframe)
    if not numeric:
        st.warning("EDA charts require at least one numerical column.")
        return
    selected = st.selectbox("Histogram and box plot column", numeric, help="Pandas selects a numerical column for Matplotlib and Seaborn visualizations.")
    st.markdown(f'{graph_tooltip("Histogram")} &nbsp;&nbsp; {graph_tooltip("Box Plot")}', unsafe_allow_html=True)
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
    render_graph_details("Histogram", "Box Plot")
    if len(numeric) >= 2:
        st.markdown(f'{graph_tooltip("Correlation Heatmap")} &nbsp;&nbsp; {graph_tooltip("Scatter Plot")} ', unsafe_allow_html=True)
        st.subheader("Correlation and relationships")
        left, right = st.columns(2)
        with left:
            figure, axis = chart_figure((6, 4))
            sns.heatmap(dataframe[numeric].corr(), annot=True, fmt=".2f", cmap="mako", ax=axis, cbar=False)
            axis.set_title("Correlation heatmap", color="white")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            render_graph_details("Correlation Heatmap")
        with right:
            x_axis, y_axis = st.columns(2)
            x_column = x_axis.selectbox("X axis", numeric, key="eda_x", help="Select the numerical feature plotted on the horizontal axis.")
            y_column = y_axis.selectbox("Y axis", [column for column in numeric if column != x_column] or numeric, key="eda_y", help="Select the numerical feature plotted on the vertical axis.")
            figure, axis = chart_figure((6, 4))
            sns.scatterplot(data=dataframe, x=x_column, y=y_column, ax=axis, color=ACCENT)
            axis.set_title(f"{x_column} vs {y_column}", color="white")
            figure.tight_layout()
            st.pyplot(figure)
            plt.close(figure)
            render_graph_details("Scatter Plot")


def render_ml(dataframe: pd.DataFrame) -> None:
    render_section_title("Machine Learning", ["Scikit-learn", "Train/Test Split", "Model Evaluation"], "Scikit-learn handles train/test splitting, classification, regression, preprocessing, and measured model evaluation.")
    if len(dataframe) < 8 or len(dataframe.columns) < 2:
        st.warning("Training needs at least 8 rows and one feature plus one target column.")
        return
    target = st.selectbox("Target column", dataframe.columns, key="ml_target")
    suggested = infer_task(dataframe[target])
    task = st.selectbox("Prediction type", ["Automatic", "Classification", "Regression"], index=0, help="Scikit-learn trains a classification model for categories or a regression model for continuous numbers.")
    resolved_task = suggested if task == "Automatic" else task
    st.caption(f"Selected task: {resolved_task}. Models are evaluated on a held-out test split with random_state=42.")
    st.markdown(info_tooltip("Train/Test Split", "Separates training data used to learn patterns from testing data used to evaluate unseen data.", "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)", "Scikit-learn"), unsafe_allow_html=True)
    st.markdown(info_tooltip("Machine Learning", "Trains a model using historical data and evaluates its performance.", "X = df[features]\ny = df[target]\nmodel.fit(X_train, y_train)", "Scikit-learn"), unsafe_allow_html=True)
    if st.button("Train models", type="primary", help="Fit the selected Scikit-learn models on the active dataset."):
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
        if resolved_task == "Classification":
            metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
        else:
            metric_names = ["MAE", "MSE", "RMSE", "R2"]
        for metric_name in metric_names:
            purpose, code, library = METRIC_TOOLTIPS[metric_name]
            render_code_button(f"metric_{metric_name}", metric_name, code, library, purpose)
        st.dataframe(artifact["results"].style.format({column: "{:.3f}" for column in artifact["results"].columns if column != "Model"}), use_container_width=True, hide_index=True)
        if resolved_task == "Classification":
            render_code_button("classification", "Classification", "model = LogisticRegression()\nmodel.fit(X_train, y_train)", "Scikit-learn", "Predicts categories or classes such as Churn Yes or No.")
            render_code_button("random_forest_classifier", "Random Forest", "model = RandomForestClassifier(random_state=42)", "Scikit-learn", "Combines multiple decision trees for classification.")
        else:
            render_code_button("linear_regression", "Linear Regression", "model = LinearRegression()\nmodel.fit(X_train, y_train)\nprediction = model.predict(X_test)", "Scikit-learn", "Predicts a continuous numerical target by learning relationships between inputs and outputs.")
            render_code_button("random_forest_regressor", "Random Forest", "model = RandomForestRegressor(random_state=42)", "Scikit-learn", "Combines multiple decision trees for regression.")
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
            st.markdown(graph_tooltip("Confusion Matrix"), unsafe_allow_html=True)
            st.pyplot(figure)
            plt.close(figure)
            render_graph_details("Confusion Matrix")


def render_clustering(dataframe: pd.DataFrame) -> None:
    render_section_title("Clustering", ["K-Means", "StandardScaler"], "Numerical features are scaled with StandardScaler before K-Means groups similar records.")
    numeric = numeric_columns(dataframe)
    if len(numeric) < 2:
        st.warning("K-Means visualization needs at least two numerical feature columns.")
        return
    selected = st.multiselect("Numerical features", numeric, default=numeric[: min(3, len(numeric))], help="Select numerical features for StandardScaler and K-Means.")
    clusters = st.slider("Number of clusters", min_value=2, max_value=min(8, max(2, len(dataframe) - 1)), value=3, help="Choose how many groups K-Means should create.")
    render_code_button("standard_scaler", "StandardScaler", "scaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)", "Scikit-learn", "Scales numerical features to comparable ranges before distance-based K-Means.")
    render_code_button("kmeans", "K-Means Clustering", "model = KMeans(n_clusters=3, random_state=42, n_init=10)\nclusters = model.fit_predict(X)", "Scikit-learn", "Groups similar observations into clusters.")
    if st.button("Train K-Means", type="primary", help="Scale the selected features and fit a K-Means clustering model."):
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
        st.markdown(graph_tooltip("Cluster Visualization"), unsafe_allow_html=True)
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
        render_graph_details("Cluster Visualization")
        st.dataframe(clustered, use_container_width=True)
        render_code_button("clustered_download", "Download clustered dataset", "csv = clustered_df.to_csv(index=False)\nst.download_button(\"Download\", csv)", "Pandas + Streamlit", "Saves the dataset with its K-Means cluster labels.")
        st.download_button("Download clustered dataset", clustered.to_csv(index=False).encode("utf-8"), "insightforge_clustered.csv", "text/csv", help="Pandas + Streamlit: save the dataset with its K-Means cluster labels.")


def render_prediction() -> None:
    render_section_title("Prediction", ["Scikit-learn"], "User inputs are passed to a trained Scikit-learn pipeline to generate a real prediction and, for classification, model probabilities when available.")
    artifact = st.session_state.get("model_artifact")
    if not artifact:
        st.info("Train a model in Machine Learning before making a prediction.")
        return
    model_name = st.session_state.get("selected_model", next(iter(artifact["models"])))
    model = artifact["models"].get(model_name, next(iter(artifact["models"].values())))
    render_code_button("model_prediction", "Model Prediction", "prediction = model.predict(new_data)", "Scikit-learn", "Uses a trained Machine Learning model to predict an output for new input data.")
    st.caption(f"Model: {model_name} | Target: {artifact['target']}")
    input_values: dict[str, Any] = {}
    for feature in artifact["features"].columns:
        series = artifact["features"][feature]
        if pd.api.types.is_numeric_dtype(series):
            median = float(pd.to_numeric(series, errors="coerce").median())
            render_code_button(f"prediction_{feature}", "Model Feature", f'new_data = pd.DataFrame({{{feature!r}: [value]}})', "Pandas + Scikit-learn", "A feature value supplied to the trained Machine Learning model.")
            input_values[feature] = st.number_input(feature, value=median if np.isfinite(median) else 0.0, help="This value becomes an input feature for the trained model.")
        else:
            options = series.dropna().astype(str).unique().tolist()
            render_code_button(f"prediction_{feature}", "Model Feature", f'new_data = pd.DataFrame({{{feature!r}: [value]}})', "Pandas + Scikit-learn", "A feature value supplied to the trained Machine Learning model.")
            input_values[feature] = st.selectbox(feature, options or ["Unknown"], help="This value becomes an input feature for the trained model.")
    if st.button("Predict", type="primary", help="Pass the entered feature values to the trained model."):
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
        render_code_button("prediction_download", "Download prediction result", "csv = pd.DataFrame([result]).to_csv(index=False)\nst.download_button(\"Download\", csv)", "Pandas + Streamlit", "Saves the prediction result as a CSV file.")
        st.download_button("Download prediction result", pd.DataFrame([result]).to_csv(index=False).encode("utf-8"), "insightforge_prediction.csv", "text/csv", help="Pandas + Streamlit: save the prediction result as CSV.")


def workflow_step(label: str, technology: str, code: str, purpose: str) -> str:
    return f'<span class="workflow-step">{escape(label)}<span class="tooltip-card"><span class="tooltip-title">{escape(technology)}</span><code class="tooltip-code">{escape(code)}</code><span class="tooltip-purpose">{escape(purpose)}</span></span></span>'


def render_about() -> None:
    render_section_title("About DATA WATCH", ["Python", "Pandas", "NumPy", "Scikit-learn", "Matplotlib", "Seaborn", "Streamlit"], "The complete technology stack powers the data, visualization, Machine Learning, and dashboard workflow.")
    render_technology_stack()
    st.subheader("How It Works")
    workflow = [
        workflow_step("CSV Upload", "Pandas", "pd.read_csv('data.csv')", "Loads the selected CSV into a DataFrame."),
        workflow_step("Pandas DataFrame", "Pandas", "dataframe.head()", "Provides the table used by every section."),
        workflow_step("Data Cleaning", "Pandas + NumPy", "dataframe.isna().sum()", "Checks missing values and prepares usable data."),
        workflow_step("EDA", "Matplotlib + Seaborn", "sns.heatmap(dataframe.corr())", "Visualizes distributions and relationships."),
        workflow_step("Feature Processing", "Scikit-learn", "StandardScaler().fit_transform(features)", "Imputes, encodes, and scales model features."),
        workflow_step("Machine Learning", "Scikit-learn", "model.fit(X_train, y_train)", "Trains classification or regression models."),
        workflow_step("Model Evaluation", "Scikit-learn", "r2_score(y_test, predictions)", "Measures performance on held-out data."),
        workflow_step("Prediction", "Scikit-learn", "model.predict(new_data)", "Generates a prediction from user inputs."),
    ]
    workflow_html = '<span class="workflow-arrow">&darr;</span>'.join(workflow)
    st.markdown(f'<div class="workflow">{workflow_html}</div>', unsafe_allow_html=True)
    st.markdown("""
    DATA WATCH is a portfolio-ready analytics workspace for understanding tabular datasets and testing practical baseline machine-learning models.

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
        st.session_state["original_dataset"] = sample_data.copy() if sample_data is not None else None
        st.session_state["dataset_source"] = sample_source

    with st.sidebar:
        st.markdown('<div class="brand"><div class="brand-name"><span class="brand-mark">DW</span> DATA WATCH</div><div class="brand-subtitle">Using Machine Learning</div></div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"], help="Streamlit receives the file; Pandas reads it into a DataFrame for analysis and Machine Learning.")
        render_code_button("csv_upload", "CSV Dataset Upload", "uploaded_file = st.file_uploader(\"Upload CSV\", type=[\"csv\"])\ndf = pd.read_csv(uploaded_file)", "Streamlit + Pandas", "Uploads and loads a CSV dataset for analysis and Machine Learning.")
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
                    st.session_state["original_dataset"] = uploaded_data.copy()
                    st.session_state["dataset_source"] = uploaded_source
                    st.session_state["editor_version"] = st.session_state.get("editor_version", 0) + 1
                    clear_dataset_artifacts()
                    st.session_state.pop("prediction_result", None)
                    st.rerun()
        st.divider()
        page = st.radio("Navigate", ["Dashboard", "Dataset Analyzer", "Dataset Editor", "Exploratory Data Analysis", "Machine Learning", "Clustering", "Prediction", "About"], label_visibility="collapsed")

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
    elif page == "Dataset Editor":
        render_editor(dataframe)
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
