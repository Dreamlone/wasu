from pathlib import Path
import matplotlib.pyplot as plt

import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def visualize_time_series_data():
    train_df = pd.read_csv(Path('../data/train.csv'), parse_dates=['year'])
    train_df = train_df.dropna()

    test_monthly_df = pd.read_csv(Path('../data/test_monthly_naturalized_flow.csv'), parse_dates=['year'])
    train_monthly_df = pd.read_csv(Path('../data/train_monthly_naturalized_flow.csv'), parse_dates=['year'])

    for site in list(train_df['site_id'].unique()):
        site_df = train_df[train_df['site_id'] == site]
        site_df = site_df.sort_values(by='year')

        plt.plot(site_df['year'], site_df['volume'])
        plt.title(site)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel('Volume')
        plt.show()


if __name__ == '__main__':
    visualize_time_series_data()
