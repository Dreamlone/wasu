from pathlib import Path
import matplotlib.pyplot as plt

import warnings

import pandas as pd

from wasu.development.paths import path_to_plots_folder

warnings.filterwarnings('ignore')


def visualize_time_series_data():
    """ NB: interpretation of values is not very precise but for initial visualization is enough """
    plots_folder = Path(path_to_plots_folder(), 'basic_time_series_visualizations')
    plots_folder.mkdir(exist_ok=True, parents=True)

    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    train_df = train_df.dropna()

    parse_dates = ['forecast_year']
    test_monthly_df = pd.read_csv(Path('../../data/test_monthly_naturalized_flow.csv'), parse_dates=parse_dates)
    train_monthly_df = pd.read_csv(Path('../../data/train_monthly_naturalized_flow.csv'), parse_dates=parse_dates)
    monthly_df = pd.concat([train_monthly_df, test_monthly_df])
    monthly_df = monthly_df.sort_values(by=['site_id', 'forecast_year', 'year'])

    monthly_df['date whose month the total streamflow value is for'] = (monthly_df['year'].astype(str) + ' ' + monthly_df['month'].astype(str))
    monthly_df['date whose month the total streamflow value is for'] = pd.to_datetime(monthly_df['date whose month the total streamflow value is for'])

    for site in list(train_df['site_id'].unique()):
        site_df = train_df[train_df['site_id'] == site]
        site_df = site_df.sort_values(by='year')

        monthly_site_df = monthly_df[monthly_df['site_id'] == site]
        if monthly_site_df.empty:
            # Skip current site
            continue
        monthly_site_df = monthly_site_df.dropna()
        cumulative = monthly_site_df.groupby(['year']).agg({'volume': 'sum'}).reset_index()

        fig_size = (17.0, 6.0)
        fig, ax = plt.subplots(figsize=fig_size)
        plt.plot(site_df['year'], site_df['volume'], '-ok', color='blue', label='Train volume (seasonal)')
        plt.plot(monthly_site_df['date whose month the total streamflow value is for'],
                 monthly_site_df['volume'], color='red', label='Antecedent monthly naturalized flow')
        plt.plot(pd.to_datetime(cumulative['year'], format='%Y'), cumulative['volume'], color='orange',
                 label='Cumulative sum of antecedent monthly naturalized flow')
        plt.legend()
        plt.title(site)
        plt.grid()
        plt.xlabel('Year')
        plt.ylabel('Volume')

        plt.savefig(Path(plots_folder, f'ts_{site}.png'))
        plt.close()


if __name__ == '__main__':
    visualize_time_series_data()
