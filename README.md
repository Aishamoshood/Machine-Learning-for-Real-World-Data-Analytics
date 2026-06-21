# Adult Income Classification - COEN807

This project implements an end-to-end supervised machine learning workflow for predicting whether an individual's annual income exceeds $50K using the UCI Adult dataset.

## Dataset
Official source:
1. https://archive-beta.ics.uci.edu/dataset/2/adult
2. https://archive.ics.uci.edu/ml/datasets/census+income

## Project structure
1. `src/config.py` - project paths, constants, directory setup
2. `src/data_loader.py` - dataset fetching and raw data export
3. `src/preprocessing.py` - data cleaning and preprocessing pipeline
4. `src/feature_engineering.py` - engineered features
5. `src/eda.py` - summary outputs and visualizations
6. `src/models.py` - baseline models and tuning grids
7. `src/evaluation.py` - metrics, plots, classification reports, model saving
8. `src/experiments.py` - baseline and tuned experiment execution
9. `src/main.py` - full pipeline orchestration
10. `src/fast_experiments.py` - lightweight benchmarking for report-ready tables
11. `run_all.py` - entry point for the complete pipeline
12. `run_fast_report_experiments.py` - entry point for the lightweight report experiment
13. `run_tuning_only.py` - entry point for tuning using processed data

## Models
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting

## What the pipeline does
1. Fetches the official dataset from UCI.
2. Cleans the data and handles missing values.
3. Performs feature engineering.
4. Runs baseline models.
5. Performs hyperparameter tuning.
6. Saves trained models, plots, reports, and report-ready results tables.

## Install
```bash
pip install -r requirements.txt
```

## Run the full pipeline
```bash
python run_all.py
```

## Run the lighter report experiment
```bash
python run_fast_report_experiments.py
```

## Outputs generated
1. `data/raw/adult_raw.csv`
2. `data/processed/adult_processed.csv`
3. `outputs/eda_summary.json`
4. `outputs/baseline_metrics.csv`
5. `outputs/tuned_metrics.csv`
6. `outputs/all_metrics.csv`
7. `outputs/report_ready_table.csv`
8. `outputs/report_results_table.md`
9. `outputs/fast_report_ready_table.csv`
10. `outputs/fast_report_results_table.md`
11. `outputs/figures/confusion_matrices/`
12. `outputs/figures/roc_curves/`
13. `outputs/figures/feature_importance/`
14. `models/*.joblib`

## Reproducibility
1. Random seed fixed at 42
2. Processing and modelling implemented through consistent pipelines
3. Metrics and figures are automatically saved for report insertion

