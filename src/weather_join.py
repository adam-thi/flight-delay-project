import pandas as pd


def join_weather_to_flights(
    flights: pd.DataFrame, weather: pd.DataFrame, tolerance_hours: int = 2
) -> pd.DataFrame:
    joined_parts = []
    tolerance = pd.Timedelta(hours=tolerance_hours)

    for station in sorted(set(flights["Origin"]).intersection(set(weather["station"]))):
        flight_part = flights[flights["Origin"] == station].sort_values("scheduled_departure_utc").copy()
        weather_part = weather[weather["station"] == station].sort_values("valid").copy()

        merged = pd.merge_asof(
            flight_part,
            weather_part,
            left_on="scheduled_departure_utc",
            right_on="valid",
            direction="backward",
            tolerance=tolerance,
        )
        joined_parts.append(merged)

    joined = pd.concat(joined_parts, ignore_index=True)
    joined["weather_report_age_minutes"] = (
        joined["scheduled_departure_utc"] - joined["valid"]
    ).dt.total_seconds().div(60)
    return joined
