import pandas as pd


def prepare_flights(
    df: pd.DataFrame, sample_size: int | None = None, random_state: int = 42
) -> pd.DataFrame:
    flights = df[df["Cancelled"] == 0].copy()
    flights = flights.dropna(
        subset=[
            "FlightDate",
            "CRSDepTime",
            "DepDelay",
            "Reporting_Airline",
            "Origin",
            "Dest",
        ]
    )
    if sample_size is not None:
        flights = flights.sample(n=sample_size, random_state=random_state)

    flights["Delay"] = (flights["DepDelay"] > 15).astype(int)
    flights = flights.drop(columns=["DepDelay"])

    flights["FlightDate"] = pd.to_datetime(flights["FlightDate"])
    flights["CRSDepTime"] = flights["CRSDepTime"].astype(int)
    flights["dep_hour"] = flights["CRSDepTime"] // 100

    dep_minutes = (flights["CRSDepTime"] // 100) * 60 + (flights["CRSDepTime"] % 100)
    flights["scheduled_departure_local"] = flights["FlightDate"] + pd.to_timedelta(dep_minutes, unit="m")

    flights["route"] = flights["Origin"] + "_" + flights["Dest"]
    flights["is_weekend"] = flights["DayOfWeek"].isin([6, 7]).astype(int)
    flights["time_of_day_bin"] = pd.cut(
        flights["dep_hour"],
        bins=[-1, 5, 11, 16, 20, 23],
        labels=["overnight", "morning", "afternoon", "evening", "night"],
    )
    return flights


def add_scheduled_departure_utc(flights: pd.DataFrame, timezone_map: dict[str, str]) -> pd.DataFrame:
    pieces = []
    for airport, group in flights.groupby("Origin", group_keys=False):
        tz_name = timezone_map.get(airport)
        if tz_name is None:
            raise ValueError(f"Missing timezone for airport: {airport}")

        group = group.copy()
        localized = group["scheduled_departure_local"].dt.tz_localize(
            tz_name,
            nonexistent="shift_forward",
            ambiguous="NaT",
        )
        group["scheduled_departure_utc"] = localized.dt.tz_convert("UTC").dt.tz_localize(None)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def add_delay_rate_feature(
    train_data: pd.DataFrame, test_data: pd.DataFrame, group_col: str, target_col: str, new_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_rate = train_data[target_col].mean()
    rates = train_data.groupby(group_col)[target_col].mean()

    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data[new_col] = train_data[group_col].map(rates).fillna(global_rate)
    test_data[new_col] = test_data[group_col].map(rates).fillna(global_rate)
    return train_data, test_data


def add_rolling_delay_features(
    df: pd.DataFrame, group_cols: list[str], prefix: str, window: str = "3h"
) -> pd.DataFrame:
    parts = []
    for _, group in df.groupby(group_cols, group_keys=False):
        group = group.sort_values("scheduled_departure_utc").copy()
        ts = group.set_index("scheduled_departure_utc")
        delay_shifted = ts["Delay"].shift(1)

        ts[f"{prefix}_delay_rate_prev_3h"] = delay_shifted.rolling(window, min_periods=1).mean()
        ts[f"{prefix}_delay_count_prev_3h"] = delay_shifted.rolling(window, min_periods=1).sum()
        ts[f"{prefix}_flight_count_prev_3h"] = delay_shifted.rolling(window, min_periods=1).count()

        ts = ts.reset_index()
        parts.append(ts)

    return pd.concat(parts, ignore_index=True)


def prepare_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df.copy()
    model_df["gust"] = model_df["gust"].fillna(0)
    model_df["p01i"] = model_df["p01i"].fillna(0)
    model_df["vsby"] = model_df["vsby"].fillna(10)
    model_df["wxcodes"] = model_df["wxcodes"].fillna("")
    model_df["skyc1"] = model_df["skyc1"].fillna("CLR")

    model_df["has_precip"] = (model_df["p01i"] > 0).astype(int)
    model_df["low_visibility"] = (model_df["vsby"] < 3).astype(int)
    model_df["high_wind"] = (model_df["sknt"].fillna(0) >= 15).astype(int)
    model_df["has_gust"] = (model_df["gust"] > 0).astype(int)
    model_df["has_weather_code"] = (model_df["wxcodes"] != "").astype(int)

    required_weather_cols = ["valid", "tmpf", "relh", "sknt", "alti", "weather_report_age_minutes"]
    return model_df.dropna(subset=required_weather_cols).copy()
