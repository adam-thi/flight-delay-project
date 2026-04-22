from pathlib import Path

import pandas as pd

from src.config import FLIGHT_DATA_DIR, FLIGHT_FILE_GLOB, WEATHER_DATA_DIR, WEATHER_FILE_GLOB


FLIGHT_COLUMNS = [
    "Month",
    "FlightDate",
    "DayOfWeek",
    "Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepDelay",
    "Cancelled",
]


def load_flights(raw_data_dir: Path | None = None) -> pd.DataFrame:
    data_dir = raw_data_dir or FLIGHT_DATA_DIR
    csv_files = sorted(data_dir.glob(FLIGHT_FILE_GLOB))
    parts = [pd.read_csv(file, usecols=FLIGHT_COLUMNS) for file in csv_files]
    return pd.concat(parts, ignore_index=True)


def load_weather(weather_dir: Path | None = None) -> pd.DataFrame:
    data_dir = weather_dir or WEATHER_DATA_DIR
    csv_files = sorted(
        path for path in data_dir.glob(WEATHER_FILE_GLOB) if "2022_01_to_2022_03" not in path.name
    )
    parts = []
    for path in csv_files:
        weather_part = pd.read_csv(path)
        weather_part["valid"] = pd.to_datetime(weather_part["valid"])
        parts.append(weather_part)
    return pd.concat(parts, ignore_index=True)
