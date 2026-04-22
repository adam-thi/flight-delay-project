from datetime import datetime
import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

from src.config import (
    AIRPORT_TIMEZONES,
    OUTPUT_FIGURES_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_REPORTS_DIR,
    TEST_YEAR,
    TRAIN_YEAR,
)
from src.data import load_flights, load_weather
from src.features import (
    add_rolling_delay_features,
    add_scheduled_departure_utc,
    prepare_flights,
    prepare_weather_features,
)
from src.models import build_random_forest_pipeline, evaluate_predictions
from src.weather_join import join_weather_to_flights


def save_outputs(
    model,
    metrics: dict[str, float],
    y_true: pd.Series,
    y_pred: pd.Series,
    *,
    run_config: dict,
    dataset_summary: dict,
) -> None:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_dir = OUTPUT_MODELS_DIR / run_id
    report_dir = OUTPUT_REPORTS_DIR / run_id
    figure_dir = OUTPUT_FIGURES_DIR / run_id

    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_dir / "random_forest_pipeline.joblib")

    metrics_payload = {
        "run_id": run_id,
        "train_year": TRAIN_YEAR,
        "test_year": TEST_YEAR,
        **metrics,
    }
    with (report_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    pd.DataFrame([metrics_payload]).to_csv(report_dir / "metrics.csv", index=False)

    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    with (report_dir / "classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    with (report_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, zero_division=0))

    with (report_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    with (report_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, indent=2)

    cm = pd.crosstab(
        pd.Series(y_true, name="actual"),
        pd.Series(y_pred, name="predicted"),
        dropna=False,
    )
    cm.to_csv(report_dir / "confusion_matrix.csv")

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(report_dir / "feature_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_xticks(range(len(cm.columns)))
    ax.set_xticklabels(cm.columns)
    ax.set_yticks(range(len(cm.index)))
    ax.set_yticklabels(cm.index)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm.iloc[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Saved outputs under outputs/*/{run_id}")


def main() -> None:
    flights = prepare_flights(load_flights())
    weather = load_weather()

    flights = flights[flights["Origin"].isin(sorted(weather["station"].dropna().unique()))].copy()
    flights = add_scheduled_departure_utc(flights, AIRPORT_TIMEZONES)
    flights = flights.dropna(subset=["scheduled_departure_utc"]).copy()

    model_df = join_weather_to_flights(flights, weather, tolerance_hours=2)
    weather_join_coverage = float(model_df["valid"].notna().mean())
    model_df = prepare_weather_features(model_df)
    model_df = model_df.sort_values("scheduled_departure_utc").copy()

    model_df = add_rolling_delay_features(model_df, ["Reporting_Airline"], "airline")
    model_df = add_rolling_delay_features(model_df, ["Reporting_Airline", "Origin"], "airline_origin")

    rolling_feature_cols = [
        "airline_delay_rate_prev_3h",
        "airline_delay_count_prev_3h",
        "airline_flight_count_prev_3h",
        "airline_origin_delay_rate_prev_3h",
        "airline_origin_delay_count_prev_3h",
        "airline_origin_flight_count_prev_3h",
    ]
    for col in rolling_feature_cols:
        model_df[col] = model_df[col].fillna(0)

    train_df = model_df[model_df["FlightDate"].dt.year == TRAIN_YEAR].copy()
    test_df = model_df[model_df["FlightDate"].dt.year == TEST_YEAR].copy()

    numeric_features = [
        "DayOfWeek",
        "dep_hour",
        "is_weekend",
        "tmpf",
        "relh",
        "sknt",
        "alti",
        "vsby",
        "weather_report_age_minutes",
        "p01i",
        "airline_delay_rate_prev_3h",
        "airline_delay_count_prev_3h",
        "airline_flight_count_prev_3h",
        "airline_origin_delay_rate_prev_3h",
        "airline_origin_delay_count_prev_3h",
        "airline_origin_flight_count_prev_3h",
    ]
    categorical_features = [
        "Reporting_Airline",
        "Origin",
        "Dest",
        "route",
        "time_of_day_bin",
    ]

    features = numeric_features + categorical_features
    model = build_random_forest_pipeline(
        numeric_features,
        categorical_features,
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=5,
    )

    model.fit(train_df[features], train_df["Delay"])
    y_pred = model.predict(test_df[features])
    metrics = evaluate_predictions(test_df["Delay"], y_pred)

    run_config = {
        "train_year": TRAIN_YEAR,
        "test_year": TEST_YEAR,
        "weather_join_tolerance_hours": 2,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_name": "RandomForestClassifier",
        "model_params": {
            "n_estimators": 150,
            "max_depth": 14,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
        },
    }
    dataset_summary = {
        "raw_flight_rows_loaded": int(len(flights)),
        "weather_rows_loaded": int(len(weather)),
        "weather_join_coverage": weather_join_coverage,
        "post_weather_feature_rows": int(len(model_df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_delay_rate": float(train_df["Delay"].mean()),
        "test_delay_rate": float(test_df["Delay"].mean()),
        "origin_airports_used": int(train_df["Origin"].nunique()),
    }

    save_outputs(
        model,
        metrics,
        test_df["Delay"],
        y_pred,
        run_config=run_config,
        dataset_summary=dataset_summary,
    )
    print(pd.Series(metrics))


if __name__ == "__main__":
    main()
