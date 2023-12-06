from pathlib import Path

import warnings

import pandas as pd

from wasu.development.data.snodas import collect_snodas_data_for_site

warnings.filterwarnings('ignore')


def load_snodas():
    df = collect_snodas_data_for_site(path_to_folder=Path('../../data/snodas'),
                                      site_id='hungry_horse_reservoir_inflow')


if __name__ == '__main__':
    load_snodas()
