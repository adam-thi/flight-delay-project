from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
FLIGHT_DATA_DIR = RAW_DATA_DIR / "flight"
WEATHER_DATA_DIR = RAW_DATA_DIR / "weather"

FLIGHT_FILE_GLOB = "**/*_flight.csv"
WEATHER_FILE_GLOB = "asos_top20_origins_20*.csv"

OUTPUTS_DIR = ROOT_DIR / "outputs"
OUTPUT_MODELS_DIR = OUTPUTS_DIR / "models"
OUTPUT_FIGURES_DIR = OUTPUTS_DIR / "figures"
OUTPUT_REPORTS_DIR = OUTPUTS_DIR / "reports"

TRAIN_YEAR = 2022
TEST_YEAR = 2023

AIRPORT_TIMEZONES = {
    "ATL": "America/New_York",
    "BOS": "America/New_York",
    "CLT": "America/New_York",
    "DCA": "America/New_York",
    "DTW": "America/New_York",
    "EWR": "America/New_York",
    "JFK": "America/New_York",
    "LGA": "America/New_York",
    "MCO": "America/New_York",
    "MIA": "America/New_York",
    "DFW": "America/Chicago",
    "IAH": "America/Chicago",
    "MSP": "America/Chicago",
    "ORD": "America/Chicago",
    "DEN": "America/Denver",
    "PHX": "America/Phoenix",
    "LAS": "America/Los_Angeles",
    "LAX": "America/Los_Angeles",
    "SEA": "America/Los_Angeles",
    "SFO": "America/Los_Angeles",
}

TOP20_WEATHER_AIRPORTS = sorted(AIRPORT_TIMEZONES.keys())
