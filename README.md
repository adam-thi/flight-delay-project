# Flight Delay Prediction Baseline

This repository currently contains a single notebook baseline for predicting whether a flight will depart more than 15 minutes late.

The baseline lives in `notebooks/01_baseline_clean.ipynb` and uses only flight data from the first three months of 2022.

## Data Used

The notebook reads three BTS monthly flight files from `data/raw/`:
i got the files from this link: https://www.transtats.bts.gov/prezip/
look for the files labeled like this On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2022_1.zip
2022_1 = january 2022 , 2022_2 = feb 2022, etc

- January 2022
- February 2022
- March 2022

Combined raw rows: 1,598,468

The target is:

delay_flag = (DepDelay > 15).astype(int)


Binary classification problem:

- `0` = not delayed
- `1` = delayed

## Cleaning And Preparation

The baseline keeps only the columns needed for the first test:

- `Month`
- `DayOfWeek`
- `Reporting_Airline`
- `Origin`
- `Dest`
- `CRSDepTime`
- `DepDelay`
- `Cancelled`

Cleaning steps:

1. Load all three monthly CSV files
2. Concatenate them into one dataframe
3. Remove cancelled flights with `Cancelled == 0`
4. Drop rows missing key training fields
5. Randomly sample 200,000 rows for faster iteration
6. Create `delay_flag`
7. Create `dep_hour = CRSDepTime // 100`

After removing cancelled flights and missing values, the notebook has 1,534,734 usable rows before sampling.

## Features Used

The model trains on six features:

- `Month`
- `DayOfWeek`
- `Reporting_Airline`
- `Origin`
- `Dest`
- `dep_hour`

These are intentionally simple pre-departure features that come directly from the flight data already in the project.

## Training Method

The notebook uses:

- `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
- a scikit-learn `ColumnTransformer`
- `SimpleImputer` for missing values
- `OneHotEncoder(handle_unknown="ignore")` for categorical columns
- `RandomForestClassifier`

Model settings:

```python
RandomForestClassifier(
    n_estimators=50,
    max_depth=12,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
```

The notebook uses `class_weight="balanced"` because delayed flights are the minority class in the sampled data.

## Reports And Outputs

The notebook currently produces:

- a `classification_report`
- a confusion matrix plot
- the class distribution of `delay_flag`

Current baseline metrics:

```text
              precision    recall  f1-score   support

           0       0.87      0.63      0.73     32129
           1       0.29      0.60      0.39      7871

    accuracy                           0.63     40000
   macro avg       0.58      0.62      0.56     40000
weighted avg       0.75      0.63      0.66     40000
```

Confusion matrix:

```text
[[20343, 11786],
 [ 3170,  4701]]
```

Class balance in the sampled dataset:

- not delayed: 80.32%
- delayed: 19.68%

## Run The Baseline

Open `notebooks/01_baseline_clean.ipynb` in Jupyter and run the notebook from top to bottom.

## Environment Setup

This project is currently meant to be run in a local Python virtual environment with Jupyter.

- Python packages are listed in `requirements.txt`
- The local virtual environment folder is `.venv/`
- The baseline is run as a notebook, not as a script
- Raw data is expected under `data/raw/`

Typical setup on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter notebook
```
