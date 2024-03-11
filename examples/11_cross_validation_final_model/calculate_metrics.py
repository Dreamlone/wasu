import numpy as np
import pandas as pd
from loguru import logger

from wasu.metrics import compute_quantile_loss


def calculate_metric():
    cross_validated = pd.read_csv('cross_validated.csv')
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

    metric_low = compute_quantile_loss(y_true=np.array(labels['volume']),
                                       y_pred=np.array(cross_validated['volume_10']), quantile=0.1)
    metric_mean = compute_quantile_loss(y_true=np.array(labels['volume']),
                                        y_pred=np.array(cross_validated['volume_50']), quantile=0.5)
    metric_high = compute_quantile_loss(y_true=np.array(labels['volume']),
                                        y_pred=np.array(cross_validated['volume_90']), quantile=0.9)
    loss_metric = (metric_low + metric_mean + metric_high) / 3
    logger.info(f'Calculated loss metric: {loss_metric}')


if __name__ == '__main__':
    calculate_metric()
