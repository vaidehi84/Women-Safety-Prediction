import os
from typing import Dict, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def build_training_pipeline(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, Pipeline]:
    feature_columns = ["state", "area", "crime_type", "victim_gender", "year", "month"]
    X = data[feature_columns]
    y = data["severity_category"]

    categorical_features = ["state", "area", "crime_type", "victim_gender"]
    numeric_features = ["year", "month"]

    transformer = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("transformer", transformer),
            ("classifier", RandomForestClassifier(random_state=42, n_estimators=100)),
        ]
    )

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    return X, y_encoded, label_encoder, pipeline


def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=150),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=8),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42)
    except Exception:
        pass

    trained_models = {}
    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                (
                    "transformer",
                    ColumnTransformer(
                        transformers=[
                            (
                                "cat",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                ["state", "area", "crime_type", "victim_gender"],
                            ),
                            ("num", StandardScaler(), ["year", "month"]),
                        ],
                        remainder="drop",
                    ),
                ),
                ("classifier", model),
            ]
        )
        pipeline.fit(X, y)
        trained_models[name] = pipeline
    return trained_models


def select_best_model(models: Dict[str, object], X: pd.DataFrame, y: pd.Series) -> Tuple[str, object, Dict[str, float]]:
    metrics = {}
    best_score = -1.0
    best_model = None
    best_name = ""

    for name, model in models.items():
        score = model.score(X, y)
        metrics[name] = score
        if score > best_score:
            best_score = score
            best_model = model.named_steps["classifier"]
            best_name = name

    return best_name, best_model, metrics


def save_model(model: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> object:
    return joblib.load(path)


def build_feature_options(data: pd.DataFrame) -> Dict[str, list]:
    return {
        "state": sorted(data["state"].dropna().unique().tolist()),
        "area": sorted(data["area"].dropna().unique().tolist()),
        "crime_type": sorted(data["crime_type"].dropna().unique().tolist()),
        "victim_gender": sorted(data["victim_gender"].dropna().unique().tolist()),
        "year": sorted(data["year"].dropna().astype(int).unique().tolist()),
        "month": sorted(data["month"].dropna().astype(int).unique().tolist()),
    }


def create_prediction_record(
    state: str,
    district: str,
    crime_type: str,
    victim_gender: str,
    year: int,
    month: int,
) -> pd.DataFrame:
    record = pd.DataFrame(
        [
            {
                "state": state.title(),
                "area": district.title(),
                "crime_type": crime_type.title(),
                "victim_gender": victim_gender.title(),
                "year": int(year),
                "month": int(month),
            }
        ]
    )
    return record


def calculate_safety_score(record: pd.DataFrame, prediction: str, confidence: float) -> int:
    base = 100
    penalty = 0
    if prediction == "High":
        penalty = 45
    elif prediction == "Medium":
        penalty = 25
    elif prediction == "Low":
        penalty = 10
    score = base - penalty - int((1.0 - confidence) * 25)
    return max(15, min(100, score))


def severity_category(value: str) -> str:
    return str(value)


def get_hotspot_areas(data: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    hotspot = (
        data.groupby(["state", "area"])["cases_reported"].sum()
        .reset_index()
        .sort_values(by="cases_reported", ascending=False)
    )
    return hotspot.head(top_n)
