import os
import numpy as np
import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        str(col)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        for col in df.columns
    ]
    return df


def load_crime_data(data_path: str) -> pd.DataFrame:
    sample_path = os.path.join("dataset", "sample_crime_data.csv")
    women_crime_path = os.path.join("dataset", "crime_against_women.csv")
    dfs = []

    # Load main dataset if exists
    if os.path.exists(data_path):
        dfs.append(pd.read_csv(data_path))

    # Load additional women crime dataset if exists
    if os.path.exists(women_crime_path):
        dfs.append(pd.read_csv(women_crime_path))

    # Load sample if no datasets found
    if not dfs and os.path.exists(sample_path):
        dfs.append(pd.read_csv(sample_path))

    if not dfs:
        raise FileNotFoundError(
            "No crime datasets found. Please add dataset/crime_in_india.csv, dataset/crime_against_women.csv, or sample_crime_data.csv."
        )

    # Concatenate all available datasets
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def extract_month(df: pd.DataFrame) -> pd.Series:
    if "month" in df.columns:
        return pd.to_numeric(df["month"], errors="coerce").fillna(1).astype(int)
    if "date" in df.columns:
        try:
            return pd.to_datetime(df["date"], errors="coerce").dt.month.fillna(1).astype(int)
        except Exception:
            return pd.Series(np.ones(len(df)), index=df.index, dtype=int)
    return pd.Series(np.ones(len(df)), index=df.index, dtype=int)


def map_severity(row: pd.Series) -> str:
    text = str(row.get("crime_type", "")).lower()
    if any(keyword in text for keyword in ["murder", "rape", "kidnap", "assault"]):
        return "High"
    if any(keyword in text for keyword in ["burglary", "robbery", "theft", "dacoity"]):
        return "Medium"
    if any(keyword in text for keyword in ["dowry", "eve teasing", "harassment", "molestation"]):
        return "High"
    return "Low"


def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(raw_df)

    field_map = {
        "state_ut": "state",
        "state": "state",
        "district": "district",
        "city": "district",
        "crime_head": "crime_type",
        "crime_type": "crime_type",
        "victim_gender": "victim_gender",
        "gender": "victim_gender",
        "cases_reported": "cases_reported",
        "cases": "cases_reported",
        "number_of_cases": "cases_reported",
        "year": "year",
        "month": "month",
        "date": "date",
    }

    normalized = {}
    for source, target in field_map.items():
        if source in df.columns and target not in normalized:
            normalized[target] = df[source]

    processed = pd.DataFrame()
    processed["state"] = normalized.get("state", pd.Series(["Unknown"] * len(df)))
    processed["district"] = normalized.get("district", pd.Series(["Unknown"] * len(df)))
    processed["crime_type"] = normalized.get("crime_type", pd.Series(["Unknown"] * len(df)))
    processed["victim_gender"] = normalized.get("victim_gender", pd.Series(["Unknown"] * len(df)))
    processed["cases_reported"] = pd.to_numeric(
        normalized.get("cases_reported", pd.Series([1] * len(df))), errors="coerce"
    ).fillna(1).astype(int)
    processed["year"] = pd.to_numeric(
        normalized.get("year", pd.Series([2023] * len(df))), errors="coerce"
    ).fillna(2023).astype(int)
    processed["month"] = extract_month(df)

    processed["state"] = processed["state"].fillna("Unknown").astype(str).str.strip()
    processed["district"] = processed["district"].fillna("Unknown").astype(str).str.strip()
    processed["crime_type"] = processed["crime_type"].fillna("Unknown").astype(str).str.strip()
    processed["victim_gender"] = processed["victim_gender"].fillna("Unknown").astype(str).str.strip()

    processed["area"] = processed["district"].str.title()
    processed["crime_type"] = processed["crime_type"].str.title()
    processed["victim_gender"] = processed["victim_gender"].str.title()
    processed["state"] = processed["state"].str.title()

    processed["severity"] = processed.apply(map_severity, axis=1)
    processed["area_risk"] = processed["cases_reported"] * processed["severity"].map({"Low": 1, "Medium": 2, "High": 3})
    processed["severity_category"] = processed["severity"]

    processed = processed.drop_duplicates()
    processed = processed.reset_index(drop=True)
    return processed
