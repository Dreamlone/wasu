from pathlib import Path
import matplotlib.pyplot as plt

import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def generate_forecast_repeating_last_value():
    train_df = pd.read_csv(Path('../data/train.csv'), parse_dates=['year'])
    train_df = train_df.dropna()

    submission_format = pd.read_csv(Path('../data/submission_format.csv'), parse_dates=['issue_date'])
    df_to_send = []
    # For every site
    for site in list(submission_format['site_id'].unique()):
        submission_site = submission_format[submission_format['site_id'] == site]
        site_df = train_df[train_df['site_id'] == site]
        site_df = site_df.sort_values(by='year')

        last_known_value = site_df[site_df['year'].dt.year == 2004]
        predicted_income = last_known_value['volume'].values[0]
        lower_predicted_income = predicted_income - (predicted_income * 0.1)
        above_predicted_income = predicted_income + (predicted_income * 0.1)

        submission_site['volume_10'] = lower_predicted_income
        submission_site['volume_50'] = predicted_income
        submission_site['volume_90'] = above_predicted_income

        df_to_send.append(submission_site)

    df_to_send = pd.concat(df_to_send)
    submission_format['index'] = (submission_format['site_id'].astype(str) + submission_format['issue_date'].astype(str))
    df_to_send['index'] = (df_to_send['site_id'].astype(str) + df_to_send['issue_date'].astype(str))

    submit = submission_format[['index']].merge(df_to_send, on='index')
    submit = submit.drop(columns=['index'])
    submit.to_csv(f'repeating_values.csv', index=False)


if __name__ == '__main__':
    generate_forecast_repeating_last_value()
