import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, RANDOM_STATE, ensure_directories
from src.data_loader import fetch_adult_data
from src.experiments import build_final_report_tables, run_baseline_models, run_tuned_models
from src.feature_engineering import feature_engineering
from src.preprocessing import basic_cleaning, build_preprocessor


if __name__ == '__main__':
    ensure_directories()
    processed_path = DATA_DIR / 'processed' / 'adult_processed.csv'

    if processed_path.exists():
        df = pd.read_csv(processed_path)
    else:
        df = feature_engineering(basic_cleaning(fetch_adult_data()))
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)

    X = df.drop(columns=['income'])
    y = df['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(X_train)

    baseline_path = DATA_DIR.parent / 'outputs' / 'baseline_metrics.csv'
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df, _, _ = run_baseline_models(X_train, X_test, y_train, y_test, preprocessor)

    tuned_df = run_tuned_models(X_train, X_test, y_train, y_test, preprocessor)
    build_final_report_tables(baseline_df, tuned_df)
    print(tuned_df)
