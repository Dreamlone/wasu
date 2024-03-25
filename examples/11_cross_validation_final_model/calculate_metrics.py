from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from matplotlib import pyplot as plt

from wasu.development.paths import path_to_plots_folder
from wasu.metrics import compute_quantile_loss


def get_metric(labels, cross_validated):
    metric_low = compute_quantile_loss(y_true=np.array(labels['volume']),
                                       y_pred=np.array(cross_validated['volume_10']), quantile=0.1)
    metric_mean = compute_quantile_loss(y_true=np.array(labels['volume']),
                                        y_pred=np.array(cross_validated['volume_50']), quantile=0.5)
    metric_high = compute_quantile_loss(y_true=np.array(labels['volume']),
                                        y_pred=np.array(cross_validated['volume_90']), quantile=0.9)
    loss_metric = (metric_low + metric_mean + metric_high) / 3
    return loss_metric


def calculate_metric():
    cross_validated = pd.read_csv('cross_validated.csv', parse_dates=['issue_date'])
    labels = pd.read_csv("cross_validation_labels.csv")
    submission_format = pd.read_csv("cross_validation_submission_format.csv")
    submission_format.issue_date = pd.to_datetime(submission_format.issue_date)
    INDEX = ["site_id", "issue_date"]
    labels = submission_format.merge(
        labels,
        left_on=["site_id", submission_format.issue_date.dt.year],
        right_on=["site_id", "year"],
        how="left",
    ).set_index(INDEX)

    # Check particular year
    loss_metric = get_metric(labels,
                             cross_validated)
    logger.info(f'Calculated loss metric: {loss_metric}')

    labels['year'] = pd.to_datetime(labels['year'], format='%Y')
    labels = labels.reset_index()
    plots_folder = Path(path_to_plots_folder(), 'cross_validation_results')
    plots_folder.mkdir(exist_ok=True, parents=True)

    for site_id in cross_validated['site_id'].unique():
        test_site_df = cross_validated[cross_validated['site_id'] == site_id]
        actual_site_df = labels[labels['site_id'] == site_id]
        test_site_df = test_site_df.sort_values(by='issue_date')
        actual_site_df = actual_site_df.sort_values(by='issue_date')

        loss_metric = get_metric(actual_site_df, test_site_df)

        fig_size = (17.0, 6.0)
        fig, ax1 = plt.subplots(figsize=fig_size)
        plt.plot(actual_site_df['issue_date'], actual_site_df['volume'], '-ok', c='blue', label='Actual')
        plt.scatter(test_site_df['issue_date'], test_site_df['volume_10'], alpha=0.2, s=3, c='red')
        plt.scatter(test_site_df['issue_date'], test_site_df['volume_90'], alpha=0.2, s=3, c='red')
        plt.scatter(test_site_df['issue_date'], test_site_df['volume_50'], s=8, c='red', label='Prediction')
        plt.title(f'{site_id}. Metric: {loss_metric:.2f}')
        plt.grid()
        plt.legend()
        plt.savefig(Path(plots_folder, f'{site_id}.png'))
        plt.close()


if __name__ == '__main__':
    calculate_metric()
