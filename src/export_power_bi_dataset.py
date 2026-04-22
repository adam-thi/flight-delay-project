from pathlib import Path

import pandas as pd

from src.config import AIRPORT_TIMEZONES, PROCESSED_DATA_DIR, TEST_YEAR, TRAIN_YEAR
from src.data import load_flights, load_weather
from src.features import (
    add_rolling_delay_features,
    add_scheduled_departure_utc,
    prepare_flights,
    prepare_weather_features,
)
from src.weather_join import join_weather_to_flights


POWER_BI_SAMPLE_SIZE = 300_000
POWER_BI_RANDOM_STATE = 42
POWER_BI_FILENAME = "flight_delay_power_bi_final.csv"


def build_power_bi_export(sample_size: int = POWER_BI_SAMPLE_SIZE) -> pd.DataFrame:
    flights = prepare_flights(load_flights())
    weather = load_weather()

    flights = flights[flights["Origin"].isin(sorted(weather["station"].dropna().unique()))].copy()
    flights = add_scheduled_departure_utc(flights, AIRPORT_TIMEZONES)
    flights = flights.dropna(subset=["scheduled_departure_utc"]).copy()

    model_df = join_weather_to_flights(flights, weather, tolerance_hours=2)
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

    model_df["year"] = model_df["FlightDate"].dt.year
    model_df["model_split"] = model_df["year"].map(
        {
            TRAIN_YEAR: "train",
            TEST_YEAR: "test",
        }
    ).fillna("other")

    export_cols = [
        "FlightDate",
        "year",
        "model_split",
        "Month",
        "DayOfWeek",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "route",
        "CRSDepTime",
        "dep_hour",
        "is_weekend",
        "time_of_day_bin",
        "Delay",
        "tmpf",
        "relh",
        "sknt",
        "alti",
        "vsby",
        "p01i",
        "weather_report_age_minutes",
        "airline_delay_rate_prev_3h",
        "airline_delay_count_prev_3h",
        "airline_flight_count_prev_3h",
        "airline_origin_delay_rate_prev_3h",
        "airline_origin_delay_count_prev_3h",
        "airline_origin_flight_count_prev_3h",
    ]

    export_df = model_df[export_cols].copy()
    export_df["FlightDate"] = export_df["FlightDate"].dt.strftime("%Y-%m-%d")
    export_df["time_of_day_bin"] = export_df["time_of_day_bin"].astype(str)

    if sample_size and len(export_df) > sample_size:
        sampled_parts = []
        total_rows = len(export_df)
        for _, part in export_df.groupby(["year", "Delay"], dropna=False):
            target_n = min(
                len(part),
                max(1, round(sample_size * len(part) / total_rows)),
            )
            sampled_parts.append(part.sample(n=target_n, random_state=POWER_BI_RANDOM_STATE))
        export_df = pd.concat(sampled_parts, ignore_index=True)

    return export_df.sort_values(["FlightDate", "Origin", "CRSDepTime"]).reset_index(drop=True)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    export_path = PROCESSED_DATA_DIR / POWER_BI_FILENAME
    export_df = build_power_bi_export()
    export_df.to_csv(export_path, index=False)
    print(f"Saved {len(export_df):,} rows to {export_path}")


if __name__ == "__main__":
    main()
