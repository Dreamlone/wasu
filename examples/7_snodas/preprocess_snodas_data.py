from pathlib import Path

import warnings

from wasu.development.data.snodas import preprocess_snodas_data

warnings.filterwarnings('ignore')


def preprocess_snodas():
    """ Launch this script to prepare SNODAS data - unpack and convert into geotiff files """
    preprocess_snodas_data(path_to_folder=Path('../../data/snodas'),
                           folder_to_unpack_files=Path('../../data/snodas_unpacked'))


if __name__ == '__main__':
    preprocess_snodas()
