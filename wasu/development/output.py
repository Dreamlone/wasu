from pathlib import Path
from typing import Union

import pandas as pd


def get_project_path() -> Path:
    return Path(__file__).parent.parent.parent


def path_to_submission_file():
    return Path(get_project_path(), 'data', 'submission_format.csv')


class Output:
    """Class for preparing desired output from the model """

    def __init__(self, output_example: Union[Path, None] = None):
        self.output_example = output_example
        if self.output_example is not None:
            self.output_example = pd.read_csv(self.output_example)
        else:
            self.output_example = pd.read_csv(path_to_submission_file())
