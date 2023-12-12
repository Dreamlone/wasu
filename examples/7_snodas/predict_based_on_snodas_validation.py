from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snodas import SnodasRegression
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    validator = ModelValidation(folder_for_plots='snodas_validation')
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()

    # Load dataframe with metadata
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))
    path_to_snodas = Path('../../data/snodas_csv').resolve()

    model = SnodasRegression(train_df=train_df)
    predicted = model.predict(submission_format, metadata=metadata, path_to_snodas=path_to_snodas, vis=False)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path='./validation/snodas.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
