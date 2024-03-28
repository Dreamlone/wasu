from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import LeaveOneGroupOut
import datetime
import warnings

from wasu.metrics import compute_quantile_loss

warnings.filterwarnings('ignore')

from wasu.development.models.common import CommonRegression


PARAMETERS_BY_SITE = {
  "hungry_horse_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 104
  },
  "snake_r_nr_heise": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 80
  },
  "pueblo_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 80,
    "PDSI days": 128
  },
  "sweetwater_r_nr_alcova": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 80
  },
  "missouri_r_at_toston": {
    "SNOTEL short days": 22,
    "SNOTEL long days": 84,
    "PDSI days": 80
  },
  "animas_r_at_durango": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 92
  },
  "yampa_r_nr_maybell": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 124,
    "PDSI days": 80
  },
  "libby_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 100
  },
  "boise_r_nr_boise": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 80
  },
  "green_r_bl_howard_a_hanson_dam": {
    "SNOTEL short days": 22,
    "SNOTEL long days": 80,
    "PDSI days": 152
  },
  "taylor_park_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 92
  },
  "dillon_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 124,
    "PDSI days": 80
  },
  "ruedi_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 100,
    "PDSI days": 80
  },
  "fontenelle_reservoir_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 88
  },
  "weber_r_nr_oakley": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 104
  },
  "san_joaquin_river_millerton_reservoir": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 80,
    "PDSI days": 144
  },
  "merced_river_yosemite_at_pohono_bridge": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 80,
    "PDSI days": 128
  },
  "american_river_folsom_lake": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 120
  },
  "colville_r_at_kettle_falls": {
    "SNOTEL short days": 10,
    "SNOTEL long days": 124,
    "PDSI days": 152
  },
  "stehekin_r_at_stehekin": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 84,
    "PDSI days": 80
  },
  "detroit_lake_inflow": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 80,
    "PDSI days": 104
  },
  "virgin_r_at_virtin": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 124,
    "PDSI days": 80
  },
  "skagit_ross_reservoir": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 124,
    "PDSI days": 80
  },
  "boysen_reservoir_inflow": {
    "SNOTEL short days": 10,
    "SNOTEL long days": 80,
    "PDSI days": 96
  },
  "pecos_r_nr_pecos": {
    "SNOTEL short days": 22,
    "SNOTEL long days": 92,
    "PDSI days": 80
  },
  "owyhee_r_bl_owyhee_dam": {
    "SNOTEL short days": 34,
    "SNOTEL long days": 100,
    "PDSI days": 152
  }
}

PATH_TO_SNOTEL_DATA = Path(Path(__file__).parent.parent.parent, 'data', 'snotel')
PATH_TO_SNODAS_DATA = Path(Path(__file__).parent.parent.parent, 'data', 'snodas_csv')
PATH_TO_PDSI = Path(Path(__file__).parent.parent.parent, 'data', 'pdsi_csv')
METADATA_PATH = Path(Path(__file__).parent.parent.parent, 'data', 'metadata_TdPVeJC.csv')

labels = pd.read_csv("cross_validation_labels.csv")
submission_format = pd.read_csv("cross_validation_submission_format.csv")
submission_format.issue_date = pd.to_datetime(submission_format.issue_date)

# Merge with submission_format to get one row per issue date
INDEX = ["site_id", "issue_date"]
labels = submission_format.merge(
    labels,
    left_on=["site_id", submission_format.issue_date.dt.year],
    right_on=["site_id", "year"],
    how="left",
).set_index(INDEX)
full_train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])

#### CROSS-VALIDATION ####

# Initialize Leave-One-Group-Out cross-validator
# and a dictionary to keep track of scores for each fold
logo = LeaveOneGroupOut()
scores = {}

# Keep track of predictions generated for each fold to construct the final submission
all_preds = []

# Perform Leave-One-Group-Out cross-validation
metadata = pd.read_csv(METADATA_PATH)
predictions = []
for train_indices, test_indices in logo.split(labels.volume.values, groups=labels.year):
    # Split labels into train and test sets
    train_labels, test_labels = labels.iloc[train_indices], labels.iloc[test_indices]
    test_year = test_labels['year'].values[0]

    train_years = list(train_labels['year'].unique())
    logger.info(f'Fit and validate models for all sites for year: {test_year}. Train years: {train_years}')

    # Train your model(s) on train_labels
    train_labels = train_labels.reset_index()
    test_labels = test_labels.reset_index()
    for site_id in test_labels['site_id'].unique():
        test_site_df = test_labels[test_labels['site_id'] == site_id]
        test_site_df['year'] = pd.to_datetime(test_site_df['year'], format='%Y')

        full_train_site_df = full_train_df[full_train_df['site_id'] == site_id]
        full_train_site_df = full_train_site_df.dropna()
        history_site_df = full_train_site_df[full_train_site_df['year'] < test_site_df['year'].values[0]]
        post_history_site_df = full_train_site_df[full_train_site_df['year'] > test_site_df['year'].values[0]]
        train_site_df = pd.concat([history_site_df, post_history_site_df])

        # Fit the model and save the results into folder
        aggregation_days_snotel_short = PARAMETERS_BY_SITE[site_id]['SNOTEL short days']
        aggregation_days_snotel_long = PARAMETERS_BY_SITE[site_id]['SNOTEL long days']
        aggregation_days_pdsi = PARAMETERS_BY_SITE[site_id]['PDSI days']

        starting_time = datetime.datetime.now()
        model = CommonRegression(train_df=train_site_df, method='linear',
                                 aggregation_days_snotel_short=aggregation_days_snotel_short,
                                 aggregation_days_snotel_long=aggregation_days_snotel_long,
                                 aggregation_days_pdsi=aggregation_days_pdsi)

        logger.debug(f'Start fitting process for year: {test_year}. Site: {site_id}')
        model_path = model.fit_model_for_site(site_id=site_id, metadata=metadata,
                                              path_to_snotel=PATH_TO_SNOTEL_DATA,
                                              path_to_snodas=PATH_TO_SNODAS_DATA,
                                              path_to_pdsi=PATH_TO_PDSI, vis=False)
        spend_time = datetime.datetime.now() - starting_time
        time = spend_time.total_seconds()

        # Launch the prediction for test year
        predicted = model.predict(test_site_df, metadata=metadata, path_to_snotel=PATH_TO_SNOTEL_DATA,
                                  path_to_snodas=PATH_TO_SNODAS_DATA, path_to_pdsi=PATH_TO_PDSI)

        # Calculate metric
        metric_low = compute_quantile_loss(y_true=np.array(test_site_df['volume']),
                                           y_pred=np.array(predicted['volume_10']), quantile=0.1)
        metric_mean = compute_quantile_loss(y_true=np.array(test_site_df['volume']),
                                            y_pred=np.array(predicted['volume_50']), quantile=0.5)
        metric_high = compute_quantile_loss(y_true=np.array(test_site_df['volume']),
                                            y_pred=np.array(predicted['volume_90']), quantile=0.9)
        loss_metric = (metric_low + metric_mean + metric_high) / 3
        logger.info(f'Calculated loss metric: {loss_metric}')

        predicted['execution_time'] = time
        predictions.append(predicted)

# Save results as csv file
predictions = pd.concat(predictions)
labels['year'] = pd.to_datetime(labels['year'], format='%Y')
model = CommonRegression(train_df=labels, method='linear', aggregation_days_snotel_short=2,
                         aggregation_days_snotel_long=2, aggregation_days_pdsi=2)
model.save_predictions_as_submit(predictions[['site_id', 'issue_date', 'volume_10', 'volume_50', 'volume_90']],
                                 submission_format=pd.read_csv("cross_validation_submission_format.csv"),
                                 path='cross_validated_lg.csv')
