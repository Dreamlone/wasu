from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger

from wasu.development.paths import path_to_data_folder, path_to_examples_folder


def collect_usgs_streamflow_time_series_for_site(path_to_folder: Path, site_id: str) -> Union[pd.DataFrame, None]:
    """ Collect time series for all available years """
    all_files = list(path_to_folder.iterdir())
    all_files.sort()

    site_df = []
    for year_folder in all_files:
        try:
            site_year = pd.read_csv(Path(year_folder, f'{site_id}.csv'), parse_dates=['datetime'])
            site_df.append(site_year)
        except Exception as ex:
            logger.warning(f'Cannot process USGS streamflow file for site {site_id} in {year_folder} due to {ex}')

    if len(site_df) < 1:
        logger.info(f'There is no data for site {site_id}')
        return None

    site_df = pd.concat(site_df)
    site_df = site_df.sort_values(by='datetime')
    return site_df


class TimeSeriesPlot:
    """ Create plots with predicted and actual values per each site """

    def __init__(self):
        self.metadata = pd.read_csv(Path(path_to_data_folder(), 'metadata_TdPVeJC.csv'))
        self.train = pd.read_csv(Path(path_to_data_folder(), 'train.csv'), parse_dates=['year'])
        self.train = self.train.dropna()

        parse_dates = ['forecast_year', 'year']
        test_monthly_df = pd.read_csv(Path(path_to_data_folder(), 'test_monthly_naturalized_flow.csv'),
                                      parse_dates=parse_dates)
        train_monthly_df = pd.read_csv(Path(path_to_data_folder(), 'train_monthly_naturalized_flow.csv'),
                                       parse_dates=parse_dates)
        monthly_df = pd.concat([train_monthly_df, test_monthly_df])
        monthly_df = monthly_df.sort_values(by=['site_id', 'forecast_year', 'year'])
        self.monthly_df = monthly_df

        self.submission_format = pd.read_csv(Path(path_to_data_folder(), 'submission_format.csv'),
                                             parse_dates=['issue_date'])

        # Get missing years (in test years)
        all_years = list(range(2000, 2023))
        test_years = set(all_years) - set(list(self.train['year'].dt.year))
        self.test_years = pd.DataFrame({'year': list(test_years)})
        self.test_years['volume'] = 0
        self.test_years['year'] = pd.to_datetime(self.test_years['year'], format='%Y')

    def predicted_time_series(self, predicted: pd.DataFrame):
        """ Calculate actual volume per season for test and launch algorithm """
        plots_folder = Path(path_to_examples_folder(), 'predicted_plots')
        plots_folder.mkdir(exist_ok=True)

        for site in list(self.submission_format['site_id'].unique()):
            predicted_site = predicted[predicted['site_id'] == site]
            predicted_site = predicted_site.sort_values(by='issue_date')

            train_site_df, cumulative = self._obtain_data_for_site(site)

            plt.plot(pd.to_datetime(cumulative['forecast_year']), cumulative['volume'], color='green',
                     label='Naturalized flow', alpha=0.4)
            plt.plot(pd.to_datetime(train_site_df['year']), train_site_df['volume'], '-ok', color='orange',
                     label='Train sample')
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_50'], '-ok', color='blue',
                     label='Predicted volume', linewidth=2, linestyle='--')
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_10'],
                     color='blue', linewidth=1)
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_90'], color='blue',
                     linewidth=1)
            plt.scatter(self.test_years['year'], self.test_years['volume'], color='black',
                        label='Test years', s=140, alpha=0.6, marker="s")
            plt.xlim(min(self.test_years['year']) - pd.DateOffset(years=1),
                     max(self.test_years['year']) + pd.DateOffset(years=1))
            plt.legend()
            plt.title(site)
            plt.grid()
            plt.xlabel('Datetime')
            plt.ylabel('Volume')
            plt.savefig(Path(plots_folder, f'{site}_time_series_plot.png'))
            plt.close()

    def usgs_streamflow(self, path_to_folder: Union[str, Path]):
        """ Create plots actual data and USGS streamflow """
        plots_folder = Path(path_to_examples_folder(), 'usgs_streamflow_plots')
        plots_folder.mkdir(exist_ok=True)

        if isinstance(path_to_folder, str):
            path_to_folder = Path(path_to_folder)

        path_to_folder = path_to_folder.resolve()

        for site in list(self.metadata['site_id'].unique()):
            logger.debug(f'Created USGS plot for {site}')

            train_site_df, cumulative = self._obtain_data_for_site(site)

            streamflow_df = collect_usgs_streamflow_time_series_for_site(path_to_folder, site)
            if streamflow_df is None:
                continue

            fig_size = (15.0, 7.0)
            fig, ax1 = plt.subplots(figsize=fig_size)
            ax1.set_xlabel('Datetime')
            ax1.set_ylabel('Volume')
            ax1.plot(pd.to_datetime(cumulative['forecast_year']), cumulative['volume'], color='green',
                     label='Naturalized flow', alpha=0.4)
            ax1.plot(pd.to_datetime(train_site_df['year']), train_site_df['volume'], '-ok', color='orange',
                     label='Train sample')
            ax1.scatter(self.test_years['year'], self.test_years['volume'], color='black',
                        label='Test years', s=140, alpha=0.6, marker="s")
            ax1.tick_params(axis='y')
            ax1.legend()
            plt.xlim(min(streamflow_df['datetime']), max(streamflow_df['datetime']))
            plt.grid(c='#DCDCDC')

            ax2 = ax1.twinx()
            ax2.set_ylabel('USGS Streamflow')
            ax2.plot(pd.to_datetime(streamflow_df['datetime']), streamflow_df['00060_Mean'], color='blue', linestyle='--')
            plt.xlim(min(streamflow_df['datetime']), max(streamflow_df['datetime']))
            ax2.tick_params(axis='y')
            plt.title(f"USGS Streamflow for site {site}")
            plt.savefig(Path(plots_folder, f'{site}_time_series_plot.png'))
            plt.close()

    def _obtain_data_for_site(self, site: str):
        """ Prepare dataframes with actual train values """
        train_site_df = self.train[self.train['site_id'] == site]
        train_site_df = train_site_df.sort_values(by='year')
        metadata_site = self.metadata[self.metadata['site_id'] == site]
        season_start_month = metadata_site['season_start_month'].values[0]
        season_end_month = metadata_site['season_end_month'].values[0]

        monthly_site_df = self.monthly_df[self.monthly_df['site_id'] == site]
        monthly_site_df = monthly_site_df.dropna()
        monthly_site_df = monthly_site_df[monthly_site_df['month'] >= season_start_month]
        monthly_site_df = monthly_site_df[monthly_site_df['month'] <= season_end_month]
        cumulative = monthly_site_df.groupby(['forecast_year']).agg({'volume': 'sum'}).reset_index()

        return train_site_df, cumulative
