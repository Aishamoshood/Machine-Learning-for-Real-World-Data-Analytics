# COEN807 Adult Income Classification: Full Code and Results

This document contains the complete implementation files and the generated output tables for the Adult Income classification project.

## Dataset Source

1. https://archive-beta.ics.uci.edu/dataset/2/adult
2. https://archive.ics.uci.edu/ml/datasets/census+income

## README

```markdown
# Adult Income Classification - COEN807

This project implements an end-to-end supervised machine learning workflow for predicting whether an individual's annual income exceeds $50K using the UCI Adult dataset.

## Dataset
Official source:
- https://archive-beta.ics.uci.edu/dataset/2/adult
- https://archive.ics.uci.edu/ml/datasets/census+income

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

## Run
```bash
python run_all.py
```

## Outputs generated
- data/raw/adult_raw.csv
- data/processed/adult_processed.csv
- outputs/eda_summary.json
- outputs/baseline_metrics.csv
- outputs/tuned_metrics.csv
- outputs/all_metrics.csv
- outputs/report_ready_table.csv
- outputs/report_results_table.md
- outputs/figures/confusion_matrices/
- outputs/figures/roc_curves/
- outputs/figures/feature_importance/
- models/*.joblib

## Reproducibility
- Random seed fixed at 42
- Processing and modelling implemented through consistent pipelines
- Metrics and figures are automatically saved for report insertion
```

## Requirements

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
ucimlrepo
```

## Full End-to-End Pipeline (run_all.py)

```python
import json
import os
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

RANDOM_STATE = 42
BASE_DIR = Path('/home/user/adult_income_project')
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
OUTPUTS_DIR = BASE_DIR / 'outputs'
FIGURES_DIR = OUTPUTS_DIR / 'figures'
REPORTS_DIR = OUTPUTS_DIR / 'classification_reports'

for p in [DATA_DIR / 'raw', DATA_DIR / 'processed', MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR / 'confusion_matrices', FIGURES_DIR / 'roc_curves', FIGURES_DIR / 'feature_importance', REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid')


def fetch_adult_data() -> pd.DataFrame:
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df['income'] = y.values
    raw_path = DATA_DIR / 'raw' / 'adult_raw.csv'
    df.to_csv(raw_path, index=False)
    return df


def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().replace('-', '_') for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({'?': np.nan, 'nan': np.nan, 'None': np.nan})

    df['income'] = df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['has_capital_gain'] = (df['capital_gain'].fillna(0) > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'].fillna(0) > 0).astype(int)

    age_bins = [16, 25, 35, 45, 55, 100]
    age_labels = ['17-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

    hours_bins = [-1, 34, 45, 60, 200]
    hours_labels = ['part_time', 'full_time', 'overtime', 'extreme_overtime']
    df['work_intensity'] = pd.cut(df['hours_per_week'], bins=hours_bins, labels=hours_labels)

    region_map = {
        'United-States': 'North America', 'Canada': 'North America', 'Puerto-Rico': 'North America',
        'Mexico': 'Latin America', 'Cuba': 'Latin America', 'Jamaica': 'Latin America', 'Haiti': 'Latin America',
        'Dominican-Republic': 'Latin America', 'El-Salvador': 'Latin America', 'Guatemala': 'Latin America',
        'Nicaragua': 'Latin America', 'Columbia': 'Latin America', 'Ecuador': 'Latin America', 'Peru': 'Latin America',
        'Trinadad&Tobago': 'Latin America', 'Honduras': 'Latin America',
        'England': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe', 'Italy': 'Europe', 'Poland': 'Europe',
        'Portugal': 'Europe', 'Ireland': 'Europe', 'France': 'Europe', 'Hungary': 'Europe', 'Scotland': 'Europe',
        'Yugoslavia': 'Europe', 'Holand-Netherlands': 'Europe',
        'India': 'Asia', 'Japan': 'Asia', 'China': 'Asia', 'Iran': 'Asia', 'Philippines': 'Asia', 'Vietnam': 'Asia',
        'Laos': 'Asia', 'Taiwan': 'Asia', 'Thailand': 'Asia', 'Hong': 'Asia', 'Cambodia': 'Asia',
        'South': 'Other', 'Outlying-US(Guam-USVI-etc)': 'Other'
    }
    df['native_region'] = df['native_country'].map(region_map).fillna('Other')
    return df


def save_eda_outputs(df: pd.DataFrame) -> None:
    summary = {
        'shape': list(df.shape),
        'columns': df.columns.tolist(),
        'missing_values': df.isna().sum().to_dict(),
        'class_distribution': df['income'].value_counts(dropna=False).to_dict(),
        'numeric_summary': df.describe(include=[np.number]).round(3).to_dict(),
    }
    with open(OUTPUTS_DIR / 'eda_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(6, 4))
    income_counts = df['income'].value_counts().sort_index()
    labels = ['<=50K', '>50K']
    sns.barplot(x=labels, y=income_counts.values, palette='Blues_d')
    plt.title('Income Class Distribution')
    plt.xlabel('Income Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'class_distribution.png', dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x='age', hue='income', bins=30, kde=True, stat='density', common_norm=False)
    plt.title('Age Distribution by Income Class')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'age_distribution_by_income.png', dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    top_edu = df['education'].value_counts().head(10).index
    sns.countplot(data=df[df['education'].isin(top_edu)], y='education', hue='income', order=top_edu)
    plt.title('Top Education Levels by Income Class')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'education_by_income.png', dpi=200)
    plt.close()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def get_param_grids():
    return {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs'],
        },
        'Decision Tree': {
            'classifier__max_depth': [10, None],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__criterion': ['gini', 'entropy'],
        },
        'Random Forest': {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [20, None],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt'],
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [2, 3],
            'classifier__subsample': [0.8, 1.0],
        },
    }


def compute_metrics(y_true, y_pred, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'ROC-AUC': auc(fpr, tpr),
    }


def plot_confusion(y_true, y_pred, model_name: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(FIGURES_DIR / 'confusion_matrices' / f'{safe_name}.png', dpi=200)
    plt.close()


def plot_roc(y_true, y_prob, model_name: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    score = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'AUC = {score:.4f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(FIGURES_DIR / 'roc_curves' / f'{safe_name}.png', dpi=200)
    plt.close()


def save_classification_report(y_true, y_pred, model_name: str):
    report = classification_report(y_true, y_pred, digits=4)
    safe_name = model_name.lower().replace(' ', '_')
    with open(REPORTS_DIR / f'{safe_name}.txt', 'w', encoding='utf-8') as f:
        f.write(report)


def extract_feature_names(preprocessor: ColumnTransformer):
    num_features = preprocessor.transformers_[0][2]
    cat_pipeline = preprocessor.transformers_[1][1]
    cat_features = preprocessor.transformers_[1][2]
    ohe = cat_pipeline.named_steps['onehot']
    cat_names = ohe.get_feature_names_out(cat_features).tolist()
    return list(num_features) + cat_names


def save_feature_importance(model_pipeline: Pipeline, model_name: str):
    clf = model_pipeline.named_steps['classifier']
    preprocessor = model_pipeline.named_steps['preprocessor']
    feature_names = extract_feature_names(preprocessor)

    values = None
    if hasattr(clf, 'feature_importances_'):
        values = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        values = np.abs(clf.coef_[0])
    if values is None:
        return

    top_n = min(15, len(feature_names))
    imp = pd.Series(values, index=feature_names).sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=imp.values, y=imp.index, palette='viridis')
    plt.title(f'Top Feature Importance - {model_name}')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    safe_name = model_name.lower().replace(' ', '_')
    plt.savefig(FIGURES_DIR / 'feature_importance' / f'{safe_name}.png', dpi=200)
    plt.close()


def run_baseline_models(X_train, X_test, y_train, y_test, preprocessor):
    rows = []
    best_auc = -1
    best_name = None
    best_pipeline = None

    for model_name, model in get_models().items():
        start = time.time()
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = model_name
        metrics['Training Time (s)'] = time.time() - start
        metrics['Stage'] = 'Baseline'
        rows.append(metrics)

        joblib.dump(pipeline, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_baseline.joblib")
        plot_confusion(y_test, y_pred, model_name + ' Baseline')
        plot_roc(y_test, y_prob, model_name + ' Baseline')
        save_classification_report(y_test, y_pred, model_name + '_baseline')
        save_feature_importance(pipeline, model_name + ' Baseline')

        if metrics['ROC-AUC'] > best_auc:
            best_auc = metrics['ROC-AUC']
            best_name = model_name
            best_pipeline = pipeline

    baseline_df = pd.DataFrame(rows).sort_values(by='ROC-AUC', ascending=False)
    baseline_df.to_csv(OUTPUTS_DIR / 'baseline_metrics.csv', index=False)
    return baseline_df, best_name, best_pipeline


def run_tuned_models(X_train, X_test, y_train, y_test, preprocessor):
    rows = []
    grids = get_param_grids()

    if len(X_train) > 15000:
        X_search, _, y_search, _ = train_test_split(
            X_train, y_train, train_size=15000, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_search, y_search = X_train, y_train

    for model_name, model in get_models().items():
        base_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])
        start = time.time()

        if model_name in ['Random Forest', 'Gradient Boosting']:
            search = RandomizedSearchCV(
                estimator=base_pipeline,
                param_distributions=grids[model_name],
                n_iter=3,
                cv=3,
                scoring='f1',
                n_jobs=1,
                random_state=RANDOM_STATE,
                verbose=0,
            )
        else:
            search = GridSearchCV(
                estimator=base_pipeline,
                param_grid=grids[model_name],
                cv=3,
                scoring='f1',
                n_jobs=1,
                verbose=0,
            )

        search.fit(X_search, y_search)
        best_params = search.best_params_

        final_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model),
        ])
        final_pipeline.set_params(**best_params)
        final_pipeline.fit(X_train, y_train)

        y_pred = final_pipeline.predict(X_test)
        y_prob = final_pipeline.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics['Model'] = model_name
        metrics['Training Time (s)'] = time.time() - start
        metrics['Stage'] = 'Tuned'
        metrics['Best Params'] = json.dumps(best_params)
        rows.append(metrics)

        joblib.dump(final_pipeline, MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_tuned.joblib")
        plot_confusion(y_test, y_pred, model_name + ' Tuned')
        plot_roc(y_test, y_prob, model_name + ' Tuned')
        save_classification_report(y_test, y_pred, model_name + '_tuned')
        save_feature_importance(final_pipeline, model_name + ' Tuned')

    tuned_df = pd.DataFrame(rows).sort_values(by='ROC-AUC', ascending=False)
    tuned_df.to_csv(OUTPUTS_DIR / 'tuned_metrics.csv', index=False)
    return tuned_df


def build_report_tables(baseline_df: pd.DataFrame, tuned_df: pd.DataFrame):
    final_df = pd.concat([baseline_df, tuned_df], ignore_index=True)
    final_df.to_csv(OUTPUTS_DIR / 'all_metrics.csv', index=False)

    report_table = tuned_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']].copy()
    report_table = report_table.sort_values(by='ROC-AUC', ascending=False)
    report_table['Notes'] = [
        'Best tuned model' if i == 0 else 'Tuned comparison model' for i in range(len(report_table))
    ]
    report_table_rounded = report_table.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']:
        report_table_rounded[col] = report_table_rounded[col].map(lambda x: f'{x:.4f}')
    report_table_rounded.to_csv(OUTPUTS_DIR / 'report_ready_table.csv', index=False)

    md_lines = []
    md_lines.append('# Results Table for Report\n')
    md_lines.append('Use the tuned-model table below in the Results and Evaluation section of the report.\n')
    md_lines.append('| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Notes |')
    md_lines.append('|---|---:|---:|---:|---:|---:|---|')
    for _, row in report_table_rounded.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['Accuracy']} | {row['Precision']} | {row['Recall']} | {row['F1-score']} | {row['ROC-AUC']} | {row['Notes']} |"
        )
    md_lines.append('\n## Baseline vs Tuned Summary\n')
    combined = final_df[['Stage', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']].copy()
    for _, row in combined.iterrows():
        md_lines.append(
            f"- {row['Stage']}: {row['Model']} -> Accuracy={row['Accuracy']:.4f}, Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}, F1={row['F1-score']:.4f}, ROC-AUC={row['ROC-AUC']:.4f}"
        )
    with open(OUTPUTS_DIR / 'report_results_table.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))


def main():
    print('Fetching Adult dataset...')
    raw_df = fetch_adult_data()

    print('Cleaning and engineering features...')
    clean_df = basic_cleaning(raw_df)
    model_df = feature_engineering(clean_df)
    model_df.to_csv(DATA_DIR / 'processed' / 'adult_processed.csv', index=False)

    print('Saving EDA outputs...')
    save_eda_outputs(model_df)

    X = model_df.drop(columns=['income'])
    y = model_df['income']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    print('Running baseline models...')
    baseline_df, best_name, _ = run_baseline_models(X_train, X_test, y_train, y_test, preprocessor)
    print('Best baseline model:', best_name)

    print('Running tuned models...')
    tuned_df = run_tuned_models(X_train, X_test, y_train, y_test, preprocessor)

    print('Building report-ready tables...')
    build_report_tables(baseline_df, tuned_df)

    with open(OUTPUTS_DIR / 'run_complete.txt', 'w', encoding='utf-8') as f:
        f.write('Pipeline completed successfully.\n')

    print('Done. Check outputs folder for metrics, figures, and report-ready tables.')


if __name__ == '__main__':
    main()
```

## Fast Report Experiment Script (run_fast_report_experiments.py)

```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from ucimlrepo import fetch_ucirepo

BASE_DIR = Path('/home/user/adult_income_project')
OUTPUTS_DIR = BASE_DIR / 'outputs'
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
SAMPLE_SIZE = 12000


def load_and_prepare_data():
    adult = fetch_ucirepo(id=2)
    X = adult.data.features.copy()
    y = adult.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    df = X.copy()
    df['income'] = y.values

    df.columns = [str(c).strip().replace('-', '_') for c in df.columns]
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip().replace({'?': np.nan})

    df['income'] = df['income'].replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

    df['has_capital_gain'] = (df['capital_gain'].fillna(0) > 0).astype(int)
    df['has_capital_loss'] = (df['capital_loss'].fillna(0) > 0).astype(int)

    age_bins = [16, 25, 35, 45, 55, 100]
    age_labels = ['17-25', '26-35', '36-45', '46-55', '56+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, include_lowest=True)

    hours_bins = [-1, 34, 45, 60, 200]
    hours_labels = ['part_time', 'full_time', 'overtime', 'extreme_overtime']
    df['work_intensity'] = pd.cut(df['hours_per_week'], bins=hours_bins, labels=hours_labels)

    sample_df, _ = train_test_split(
        df, train_size=SAMPLE_SIZE, random_state=RANDOM_STATE, stratify=df['income']
    )
    X = sample_df.drop(columns=['income'])
    y = sample_df['income']
    return X, y


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    return ColumnTransformer([
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


def evaluate(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-score': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
    }


def main():
    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor(X_train)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1.0],
            'classifier__solver': ['lbfgs']
        },
        'Decision Tree': {
            'classifier__max_depth': [10, None],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 2]
        },
        'Random Forest': {
            'classifier__n_estimators': [100],
            'classifier__max_depth': [20, None],
            'classifier__min_samples_split': [2, 10],
            'classifier__min_samples_leaf': [1, 2]
        },
        'Gradient Boosting': {
            'classifier__n_estimators': [100],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__max_depth': [2, 3],
            'classifier__subsample': [0.8, 1.0]
        },
    }

    baseline_rows = []
    tuned_rows = []

    for model_name, model in models.items():
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        row = evaluate(y_test, pred, prob)
        row['Model'] = model_name
        row['Stage'] = 'Baseline'
        baseline_rows.append(row)

        if model_name in ['Random Forest', 'Gradient Boosting']:
            search = RandomizedSearchCV(
                pipe,
                grids[model_name],
                n_iter=3,
                cv=3,
                scoring='f1',
                n_jobs=1,
                random_state=RANDOM_STATE
            )
        else:
            search = GridSearchCV(
                pipe,
                grids[model_name],
                cv=3,
                scoring='f1',
                n_jobs=1
            )

        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        prob = best.predict_proba(X_test)[:, 1]
        row = evaluate(y_test, pred, prob)
        row['Model'] = model_name
        row['Stage'] = 'Tuned'
        row['Best Params'] = json.dumps(search.best_params_)
        tuned_rows.append(row)

    baseline_df = pd.DataFrame(baseline_rows).sort_values('ROC-AUC', ascending=False)
    tuned_df = pd.DataFrame(tuned_rows).sort_values('ROC-AUC', ascending=False)
    baseline_df.to_csv(OUTPUTS_DIR / 'fast_baseline_metrics.csv', index=False)
    tuned_df.to_csv(OUTPUTS_DIR / 'fast_tuned_metrics.csv', index=False)

    report_table = tuned_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']].copy()
    report_table['Notes'] = ['Best tuned model' if i == 0 else 'Tuned comparison model' for i in range(len(report_table))]
    report_table.to_csv(OUTPUTS_DIR / 'fast_report_ready_table.csv', index=False)

    with open(OUTPUTS_DIR / 'fast_report_results_table.md', 'w', encoding='utf-8') as f:
        f.write('# Fast Report Results Table\n\n')
        f.write('| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Notes |\n')
        f.write('|---|---:|---:|---:|---:|---:|---|\n')
        for _, row in report_table.iterrows():
            f.write(f"| {row['Model']} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1-score']:.4f} | {row['ROC-AUC']:.4f} | {row['Notes']} |\n")

    print('Baseline metrics')
    print(baseline_df)
    print('\nTuned metrics')
    print(tuned_df)


if __name__ == '__main__':
    main()
```

## Tuning-Only Script (run_tuning_only.py)

```python
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from run_all import (
    DATA_DIR,
    RANDOM_STATE,
    build_preprocessor,
    build_report_tables,
    feature_engineering,
    basic_cleaning,
    fetch_adult_data,
    run_baseline_models,
    run_tuned_models,
)


def main():
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

    baseline_path = Path('/home/user/adult_income_project/outputs/baseline_metrics.csv')
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
    else:
        baseline_df, _, _ = run_baseline_models(X_train, X_test, y_train, y_test, preprocessor)

    tuned_df = run_tuned_models(X_train, X_test, y_train, y_test, preprocessor)
    build_report_tables(baseline_df, tuned_df)
    print(tuned_df)


if __name__ == '__main__':
    main()
```

## Report-Ready Results Table

```markdown
# Fast Report Results Table

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | Notes |
|---|---:|---:|---:|---:|---:|---|
| Logistic Regression | 0.8633 | 0.7709 | 0.6098 | 0.6809 | 0.9168 | Best tuned model |
| Gradient Boosting | 0.8621 | 0.8015 | 0.5627 | 0.6612 | 0.9143 | Tuned comparison model |
| Random Forest | 0.8612 | 0.7684 | 0.6010 | 0.6745 | 0.9115 | Tuned comparison model |
| Decision Tree | 0.8517 | 0.7725 | 0.5383 | 0.6345 | 0.8860 | Tuned comparison model |
```

## Tuned Metrics CSV

```csv
Accuracy,Precision,Recall,F1-score,ROC-AUC,Model,Stage,Best Params
0.8633333333333333,0.7709251101321586,0.6097560975609756,0.6809338521400778,0.9168256809308821,Logistic Regression,Tuned,"{""classifier__C"": 1.0, ""classifier__solver"": ""lbfgs""}"
0.8620833333333333,0.8014888337468983,0.5627177700348432,0.661207778915046,0.9143250226118284,Gradient Boosting,Tuned,"{""classifier__subsample"": 1.0, ""classifier__n_estimators"": 100, ""classifier__max_depth"": 2, ""classifier__learning_rate"": 0.1}"
0.86125,0.7683741648106904,0.6010452961672473,0.6744868035190615,0.9115410008739424,Random Forest,Tuned,"{""classifier__n_estimators"": 100, ""classifier__min_samples_split"": 10, ""classifier__min_samples_leaf"": 1, ""classifier__max_depth"": null}"
0.8516666666666667,0.7725,0.5383275261324042,0.6344969199178645,0.8859557647759234,Decision Tree,Tuned,"{""classifier__max_depth"": 10, ""classifier__min_samples_leaf"": 2, ""classifier__min_samples_split"": 10}"
```

## Baseline Metrics CSV

```csv
Accuracy,Precision,Recall,F1-score,ROC-AUC,Model,Training Time (s),Stage
0.8698945644385301,0.7942636514065086,0.6159110350727117,0.6938087207901711,0.9244093277197839,Gradient Boosting,11.751158237457275,Baseline
0.8598628314054663,0.756484912652197,0.6112061591103507,0.676129642772652,0.9139360128580717,Logistic Regression,2.5950357913970947,Baseline
0.8518783908281298,0.7219730941704036,0.6197604790419161,0.6669735327963177,0.9022318417550965,Random Forest,2.8355729579925537,Baseline
0.8123656464325929,0.6044683491932147,0.6248930710008554,0.6145110410094637,0.7481214110218919,Decision Tree,0.9469101428985596,Baseline
```

