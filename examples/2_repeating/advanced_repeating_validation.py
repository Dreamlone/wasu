from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.repeating import AdvancedRepeatingTrainModel
from wasu.development.validation import ModelValidation

warnings.filterwarnings('ignore')


def validate_simple_model():
    validator = ModelValidation()
    train_df = pd.read_csv(Path('../../data/train.csv'), parse_dates=['year'])
    submission_format = validator.generate_submission_format()
    metadata = pd.read_csv(Path('../../data/metadata_TdPVeJC.csv'))

    # Define simple model to make a predictions
    model = AdvancedRepeatingTrainModel(train_df=train_df)
    predicted = model.predict(submission_format, metadata=metadata)

    validator.compare_dataframes(predicted, train_df)
    model.save_predictions_as_submit(predicted, path='./validation/advanced_repeating_val.csv',
                                     submission_format=submission_format)


if __name__ == '__main__':
    validate_simple_model()
