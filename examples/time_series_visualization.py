from pathlib import Path
import matplotlib.pyplot as plt

import warnings

import pandas as pd

warnings.filterwarnings('ignore')


def visualize_time_series_data():
    """ NB: interpretation of values is not very precise but for initial visualization is enough """
    train_df = pd.read_csv(Path('../data/train.csv'), parse_dates=['year'])
    train_df = train_df.dropna()

    test_monthly_df = pd.read_csv(Path('../data/test_monthly_naturalized_flow.csv'), parse_dates=['forecast_year',
                                                                                                  'year'])
    train_monthly_df = pd.read_csv(Path('../data/train_monthly_naturalized_flow.csv'))
    train_monthly_df['date whose month the total streamflow value is for'] = (train_monthly_df['year'].astype(str) + ' ' + train_monthly_df['month'].astype(str))
    train_monthly_df['date whose month the total streamflow value is for'] = pd.to_datetime(train_monthly_df['date whose month the total streamflow value is for'])

    for site in list(train_df['site_id'].unique()):
        site_df = train_df[train_df['site_id'] == site]
        site_df = site_df.sort_values(by='year')

        train_monthly_site_df = train_monthly_df[train_monthly_df['site_id'] == site]
        if train_monthly_site_df.empty:
            # Skip current site
            continue
        train_monthly_site_df = train_monthly_site_df.dropna()
        cumulative = train_monthly_site_df.groupby(['year']).agg({'volume': 'sum'}).reset_index()

        plt.plot(site_df['year'], site_df['volume'], color='blue', label='Train volume')
        plt.plot(train_monthly_site_df['date whose month the total streamflow value is for'],
                 train_monthly_site_df['volume'], color='red', label='Antecedent monthly naturalized flow')
        plt.plot(pd.to_datetime(cumulative['year'], format='%Y'), cumulative['volume'], color='orange',
                 label='Cumulative sum of antecedent monthly naturalized flow')
        plt.legend()
        plt.title(site)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel('Volume')
        plt.show()


if __name__ == '__main__':
    visualize_time_series_data()
