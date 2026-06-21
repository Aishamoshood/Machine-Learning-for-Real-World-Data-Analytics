from src.fast_experiments import run_report_experiments


if __name__ == '__main__':
    baseline_df, tuned_df = run_report_experiments()
    print('Baseline metrics')
    print(baseline_df)
    print('\nTuned metrics')
    print(tuned_df)
