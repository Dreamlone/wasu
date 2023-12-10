from pathlib import Path

import warnings

from wasu.development.data.snodas import unpack_snodas_data_into_geotiff

warnings.filterwarnings('ignore')


def unpack_archives_and_convert_results_into_geotiff():
    """ Launch this script to prepare SNODAS data - unpack and convert into geotiff files """
    unpack_snodas_data_into_geotiff(path_to_folder=Path('../../data/snodas'),
                                    folder_to_unpack_files=Path('../../data/snodas_unpacked'))


if __name__ == '__main__':
    unpack_archives_and_convert_results_into_geotiff()
