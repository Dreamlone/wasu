from pathlib import Path

import warnings


from wasu.development.data.snodas import prepare_snodas_data

warnings.filterwarnings('ignore')


def generate_forecast_based_on_snodas():
    prepare_snodas_data(path_to_folder=Path('../../data/snodas_unpacked'),
                        folder_for_csv=Path('../../data/snodas_csv'),
                        vis=False)


if __name__ == '__main__':
    generate_forecast_based_on_snodas()
