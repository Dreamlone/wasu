from pathlib import Path

import warnings

import pandas as pd

from wasu.development.data.snodas import collect_snodas_data_for_site
from wasu.development.models.snotel import SnotelFlowRegression
from wasu.development.models.streamflow import StreamFlowRegression
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    dataframe = collect_snodas_data_for_site(path_to_folder=Path('../../data/snodas_unpacked'),
                                             site_id='hungry_horse_reservoir_inflow')


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
