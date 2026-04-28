# Flight Delay Prediction

This project predicts whether a flight will depart more than 15 minutes late.

The target is binary:

- `0` = not delayed
- `1` = delayed

Right now the project has two parts:

- notebooks that show the step-by-step development of the model
- a `src/` pipeline that runs the strongest version on full data and saves outputs

## Data

Flight data comes from the BTS On-Time Reporting Carrier Performance dataset.

Current training and testing setup:

- train on 2022 flight data
- test on 2023 flight data
- use weather data for the top 20 origin airports

Main raw-data layout:

```text
data/
  raw/
    flight/
      2022/
        2022-01_flight.csv
        ...
        2022-12_flight.csv
      2023/
        2023-01_flight.csv
        ...
        2023-12_flight.csv
    weather/
      asos_top20_origins_2022.csv
      asos_top20_origins_2023.csv
```

## Notebook Flow

The notebooks are meant to tell the story of the project in order:

1. `01_baseline_clean.ipynb`
Baseline flight-only model on Jan-March 2022.

2. `02_feature_model_comparison.ipynb`
Time-based split and model comparison for flight-only features.

3. `03_feature_visualization.ipynb`
Feature plots and simple visuals for analysis and Power BI prep.

4. `04_weather_model_comparison.ipynb`
Adds weather data and compares flight-only vs flight+weather.

5. `05_final_prototype_random_forest.ipynb`
Refines the Random Forest setup for the notebook stage.

6. `06_project_summary.ipynb`
Short summary of the earlier notebook work.

7. `07_airline_delay_feature_experiment.ipynb`
Adds rolling airline delay features and shows the biggest improvement.

8. `08_feature_group_ablation.ipynb`
Tests which feature groups matter most.

9. `09_weather_feature_exploration.ipynb`
Tests stronger weather flags on a sample.

## Current `src/` Pipeline

The main script is:

```powershell
.\.venv\Scripts\python.exe -m src.train_random_forest
```

What it does:

1. loads all 2022 and 2023 flight files
2. removes cancelled flights and missing key rows
3. creates the delay target
4. converts scheduled departure time to local datetime, then UTC
5. joins the nearest earlier weather report
6. builds recent rolling airline delay features
7. trains on 2022 and tests on 2023
8. saves reports, figures, and the model artifact

The current `src/` version uses:

- schedule features like `dep_hour` and `DayOfWeek`
- route and airline categorical features
- raw weather features such as temperature, humidity, wind, visibility, pressure, and precipitation
- rolling airline and airline-origin delay features from the previous 3 hours

It does **not** use the older static delay-rate features anymore, because the ablation notebook showed they were not helping.

## Current Model Result

Recent full pipeline result:

- accuracy: about `0.700`
- delay precision: about `0.373`
- delay recall: about `0.621`
- delay F1: about `0.466`

This is not a perfect model, but it is a clear improvement over the early baseline.

The biggest lesson so far is:

- weather helps a little
- recent airline performance helps a lot

Notebook 9 tested stronger engineered weather features. They were useful for exploration, but they did not improve the final full-data model enough to keep in `src/`.

## Outputs

Each pipeline run creates a timestamped folder under `outputs/`.

Example saved artifacts:

- `outputs/models/<run_id>/random_forest_pipeline.joblib`
- `outputs/reports/<run_id>/metrics.csv`
- `outputs/reports/<run_id>/metrics.json`
- `outputs/reports/<run_id>/classification_report.txt`
- `outputs/reports/<run_id>/confusion_matrix.csv`
- `outputs/reports/<run_id>/feature_importance.csv`
- `outputs/figures/<run_id>/confusion_matrix.png`

Runs do not overwrite each other because each run gets a new timestamp folder.

## Environment

This project is set up for a local Python virtual environment on Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For notebooks:

```powershell
jupyter notebook
```

For the current pipeline:

```powershell
.\.venv\Scripts\python.exe -m src.train_random_forest
```

## Project Status

- the notebook sequence shows how the model evolved
- the `src/` pipeline reproduces the current best setup
- outputs are saved for reporting and presentation
