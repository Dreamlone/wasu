from pathlib import Path

import warnings

import pandas as pd

from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.streamflow import StreamFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    raise NotImplementedError()


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
