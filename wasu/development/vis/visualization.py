from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from wasu.development.paths import path_to_data_folder, path_to_examples_folder


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

    def predicted_time_series(self, predicted: pd.DataFrame):
        """ Calculate actual volume per season for test and launch algorithm """
        plots_folder = Path(path_to_examples_folder(), 'plots')
        plots_folder.mkdir(exist_ok=True)

        for site in list(self.submission_format['site_id'].unique()):
            train_site_df = self.train[self.train['site_id'] == site]
            train_site_df = train_site_df.sort_values(by='year')

            predicted_site = predicted[predicted['site_id'] == site]
            predicted_site = predicted_site.sort_values(by='issue_date')

            metadata_site = self.metadata[self.metadata['site_id'] == site]
            season_start_month = metadata_site['season_start_month'].values[0]
            season_end_month = metadata_site['season_end_month'].values[0]

            monthly_site_df = self.monthly_df[self.monthly_df['site_id'] == site]
            monthly_site_df = monthly_site_df.dropna()
            # Remain only values for the season
            monthly_site_df = monthly_site_df[monthly_site_df['month'] >= season_start_month]
            monthly_site_df = monthly_site_df[monthly_site_df['month'] <= season_end_month]
            cumulative = monthly_site_df.groupby(['forecast_year']).agg({'volume': 'sum'}).reset_index()

            plt.plot(pd.to_datetime(cumulative['forecast_year']), cumulative['volume'], color='green',
                     label='Naturalized flow', alpha=0.4)
            plt.plot(pd.to_datetime(train_site_df['year']), train_site_df['volume'], color='orange',
                     label='Train sample')
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_50'], '-ok', color='blue',
                     label='Predicted volume', linewidth=2, linestyle='--')
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_10'],
                     color='blue', linewidth=1)
            plt.plot(pd.to_datetime(predicted_site['issue_date']), predicted_site['volume_90'], color='blue',
                     linewidth=1)
            plt.legend()
            plt.title(site)
            plt.grid()
            plt.xlabel('Datetime')
            plt.ylabel('Volume')
            plt.savefig(Path(plots_folder, f'{site}_time_series_plot.png'))
            plt.close()
