from pathlib import Path

from wasu.development.data import collect_snotel_data_for_site

import warnings

warnings.filterwarnings('ignore')


def example_data_load(site_id: str = 'hungry_horse_reservoir_inflow'):
    df = collect_snotel_data_for_site(path_to_folder=Path('../../data/snotel'), site_id=site_id)

    print(df.head(2))
    print(f'Lenght of the dataset for {site_id}: {len(df)}')


if __name__ == '__main__':
    example_data_load()
