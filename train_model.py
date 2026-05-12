import os
from pathlib import Path
from utils.data_processing import load_crime_data, preprocess_data
from utils.model_utils import build_training_pipeline, train_models, select_best_model, save_model

MODEL_PATH = Path("models") / "best_crime_model.joblib"
DATA_PATH = Path("dataset") / "crime_in_india.csv"


def main():
    print("Loading dataset...")
    raw_data = load_crime_data(DATA_PATH)
    print(f"Dataset loaded: {raw_data.shape[0]} rows")

    processed = preprocess_data(raw_data)
    print(f"Processed dataset: {processed.shape[0]} rows, {processed.shape[1]} columns")

    X, y, label_encoder, pipeline = build_training_pipeline(processed)
    models = train_models(X, y)
    best_name, best_model, metrics = select_best_model(models, X, y)

    pipeline.set_params(classifier=best_model)
    pipeline.fit(X, y)
    artifact = {"pipeline": pipeline, "label_encoder": label_encoder}
    save_model(artifact, MODEL_PATH)

    print(f"Trained best model: {best_name}")
    print("Evaluation metrics:")
    for name, value in metrics.items():
        print(f"- {name}: {value:.4f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
