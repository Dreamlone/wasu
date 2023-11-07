import os
from pathlib import Path

import warnings

import pandas as pd
import requests
from loguru import logger

from wasu.development.models.train_model import AdvancedRepeatingTrainModel
from wasu.development.vis.visualization import TimeSeriesPlot

warnings.filterwarnings('ignore')


SOURCE_URL = "https://www.cpc.ncep.noaa.gov/data/indices/soi"
FILE_PATH_PARTS = ("teleconnections", "soi.txt")
DATA_ROOT = Path(os.getenv("WSFR_DATA_ROOT", Path.cwd() / ".." / "data"))


def load_soi_data():
    logger.info("Downloading SOI data...")
    response = requests.get(SOURCE_URL)
    out_file = DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with out_file.open("w") as fp:
        fp.write(response.text)
    logger.success(f"SOI data written to {out_file}")


if __name__ == '__main__':
    load_data()
