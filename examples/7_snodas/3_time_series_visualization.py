from pathlib import Path

import warnings

import pandas as pd
from matplotlib import pyplot as plt

from wasu.development.data.snotel import collect_snotel_data_for_site
from wasu.development.paths import path_to_plots_folder

warnings.filterwarnings('ignore')


def show_time_series(site_id: str):
    snotel_df = collect_snotel_data_for_site(path_to_folder=Path('../../data/snotel'), site_id=site_id,
                                             collect_only_in_basin=True)
    snotel_df = snotel_df.drop_duplicates()

    plots_folder = Path(path_to_plots_folder(), 'snodas_investigation')
    plots_folder.mkdir(exist_ok=True, parents=True)
    basic_features = ['Modeled snow water equivalent, total of snow layers']

    # Read real flow values
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    train_df = train_df.dropna()
    train_df = train_df[train_df['site_id'] == site_id]
    train_df['year'] = train_df['year'] + pd.DateOffset(months=4)

    file = Path(f'../../data/snodas_csv/{site_id}.csv')
    df = pd.read_csv(file, parse_dates=['datetime'])
    df = df.sort_values(by='datetime')

    for column in basic_features:
        for agg in ['mean']:
            column_name = f'{agg}_{column}'

            fig_size = (17.0, 6.0)
            fig, ax1 = plt.subplots(figsize=fig_size)
            ax1.set_xlabel('Datetime')
            ax1.set_ylabel(column)
            ax1.plot(df['datetime'], df[column_name], color='blue')
            ax1.scatter(snotel_df['date'], snotel_df['WTEQ_DAILY'], color='black', s=1,
                        label='SNOTEL data from several stations')
            ax1.tick_params(axis='y')
            plt.legend(loc='upper left')
            plt.grid(c='#DCDCDC')

            ax2 = ax1.twinx()
            ax2.scatter(train_df['year'], train_df['volume'], color='red')
            for months_offset in [1, 2, 3]:
                ax2.scatter(train_df['year'] + pd.DateOffset(months=months_offset), train_df['volume'], color='red')
            ax2.tick_params(axis='y')
            ax2.set_ylabel('Target (seasonal inflow)')
            plt.title(column_name, fontsize=14)
            plt.xlim(min(df['datetime']), max(df['datetime']))

            plt.savefig(Path(plots_folder, f'{site_id}_ts_{column_name}.png'))
            plt.close()


if __name__ == '__main__':
    show_time_series(site_id='animas_r_at_durango')
