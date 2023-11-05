from pathlib import Path
from typing import Union

import pandas as pd


class Output:
    """Class for preparing desired output from the model """

    def __init__(self, output_example: Union[Path, None] = None):
        self.output_example = output_example
        if self.output_example is not None:
            self.output_example = pd.read_csv(self.output_example)
