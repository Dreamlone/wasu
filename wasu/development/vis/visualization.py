from pathlib import Path
from typing import Union, Any, List

import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
import numpy as np

from wasu.development.data import collect_usgs_streamflow_time_series_for_site
from wasu.development.paths import path_to_data_folder, path_to_examples_folder, path_to_plots_folder


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
        all_years = list(range(2000, 2024))
        test_years = set(all_years) - set(list(self.train['year'].dt.year))
        self.test_years = pd.DataFrame({'year': list(test_years)})
        self.test_years['volume'] = 0
        self.test_years['year'] = pd.to_datetime(self.test_years['year'], format='%Y')

    def predicted_time_series(self, predicted: pd.DataFrame, plots_folder_name: str = 'predicted_plots'):
        """ Calculate actual volume per season for test and launch algorithm """
        plots_folder = Path(path_to_plots_folder(), plots_folder_name)
        plots_folder.mkdir(exist_ok=True, parents=True)

        for site in list(self.submission_format['site_id'].unique()):
            predicted_site = predicted[predicted['site_id'] == site]
            predicted_site = predicted_site.sort_values(by='issue_date')

            train_site_df, cumulative = self._obtain_data_for_site(site)

            fig_size = (20.0, 8.0)
            fig, ax = plt.subplots(figsize=fig_size)
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
        plots_folder = Path(path_to_plots_folder(), 'usgs_streamflow_plots')
        plots_folder.mkdir(exist_ok=True, parents=True)

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
            plt.xlim(min(self.test_years['year']) - pd.DateOffset(years=1),
                     max(self.test_years['year']) - pd.DateOffset(years=5))
            plt.grid(c='#DCDCDC')

            ax2 = ax1.twinx()
            ax2.set_ylabel('USGS Streamflow')
            ax2.plot(pd.to_datetime(streamflow_df['datetime']), streamflow_df['00060_Mean'], color='blue', linestyle='--')
            plt.xlim(min(streamflow_df['datetime']), max(streamflow_df['datetime']))
            ax2.tick_params(axis='y')
            plt.title(f"USGS Streamflow for site {site}")
            plt.xlim(min(self.test_years['year']) - pd.DateOffset(years=1),
                     max(self.test_years['year']) - pd.DateOffset(years=5))
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


def created_spatial_plot(dataframe_for_model_fitting: pd.DataFrame, reg: Any, features_columns: List[str],
                         folder_name: str, file_name: str, title: Union[str, None] = None):
    plots_folder = Path(path_to_plots_folder(), folder_name)
    plots_folder.mkdir(exist_ok=True, parents=True)

    cmap = 'coolwarm'
    x_vals = np.array(dataframe_for_model_fitting['min_value'])
    y_vals = np.array(dataframe_for_model_fitting['mean_value'])
    z_vals = np.array(dataframe_for_model_fitting['target'])

    # Generate dataframe for model predict
    generated_x_values = np.linspace(min(x_vals), max(x_vals), 150)
    df_with_features = []
    for x_value in generated_x_values:
        generated_y_values = np.linspace(min(y_vals), max(y_vals), 150)
        feature_df = pd.DataFrame({'mean_value': generated_y_values})
        feature_df['min_value'] = x_value
        df_with_features.append(feature_df)
    df_with_features = pd.concat(df_with_features)
    for feature in features_columns:
        if feature not in ['mean_value', 'min_value']:
            df_with_features[feature] = dataframe_for_model_fitting[feature].mean()

    predicted = reg.predict(df_with_features[features_columns])
    points = np.ravel(z_vals)

    fig = plt.figure(figsize=(16, 7))
    # First plot
    ax = fig.add_subplot(121, projection='3d')
    ax.scatter(np.array(df_with_features['min_value']),
               np.array(df_with_features['mean_value']),
               predicted, c=np.ravel(predicted), cmap=cmap, s=1, alpha=0.3, vmin=min(points), vmax=max(points))
    surf = ax.scatter(x_vals, y_vals, z_vals, c=points, cmap=cmap, edgecolors='black', linewidth=0.3, s=100)
    cb = fig.colorbar(surf, shrink=0.3, aspect=10)
    cb.set_label(f'Target', fontsize=12)
    ax.view_init(3, 10)
    ax.set_xlabel('min_value', fontsize=13)
    ax.set_ylabel('mean_value', fontsize=13)
    ax.set_zlabel('target', fontsize=13)

    # Second plot
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(np.array(df_with_features['min_value']),
               np.array(df_with_features['mean_value']),
               predicted, c=np.ravel(predicted), cmap=cmap, s=1, alpha=0.3, vmin=min(points), vmax=max(points))
    ax.scatter(x_vals, y_vals, z_vals, c=points, cmap=cmap, edgecolors='black', linewidth=0.3, s=100)
    ax.view_init(35, 50)
    ax.set_xlabel('min_value', fontsize=13)
    ax.set_ylabel('mean_value', fontsize=13)
    ax.set_zlabel('target', fontsize=13)
    if title is not None:
        plt.suptitle(title, fontsize=15)

    logger.debug(f'Saved 3d plot into {plots_folder} with name {file_name}')
    plt.savefig(Path(plots_folder, file_name))
    plt.close()
