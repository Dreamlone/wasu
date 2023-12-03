from pathlib import Path

from wasu.development.data import collect_snotel_data_for_site

import warnings

warnings.filterwarnings('ignore')


def example_data_load():
    collect_snotel_data_for_site(path_to_folder=Path('../data/snotel'), site_id='hungry_horse_reservoir_inflow')


if __name__ == '__main__':
    example_data_load()
