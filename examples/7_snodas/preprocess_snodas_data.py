from pathlib import Path

import warnings

import pandas as pd

from wasu.development.data.snodas import collect_snodas_data_for_site, preprocess_snodas_data

warnings.filterwarnings('ignore')


def preprocess_snodas():
    """ Launch this script to prepare SNODAS data - unpack and convert into geotiff files """
    preprocess_snodas_data(path_to_folder=Path('../../data/snodas'),
                           new_folder_name='snodas_unpacked')


if __name__ == '__main__':
    preprocess_snodas()
