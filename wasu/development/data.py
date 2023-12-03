from pathlib import Path
from typing import Union

import pandas as pd
from loguru import logger


def collect_usgs_streamflow_time_series_for_site(path_to_folder: Path, site_id: str) -> Union[pd.DataFrame, None]:
    """ Collect time series for all available years """
    all_files = list(path_to_folder.iterdir())
    all_files.sort()

    site_df = []
    for year_folder in all_files:
        try:
            site_year = pd.read_csv(Path(year_folder, f'{site_id}.csv'), parse_dates=['datetime'])
            site_df.append(site_year)
        except Exception as ex:
            logger.warning(f'Cannot process USGS streamflow file for site {site_id} in {year_folder} due to {ex}')

    if len(site_df) < 1:
        logger.info(f'There is no data for site {site_id}')
        return None

    site_df = pd.concat(site_df)
    site_df = site_df.sort_values(by='datetime')
    return site_df


def collect_snotel_data_for_site(path_to_folder: Path, site_id: str):
    """ Load SNOTEL data from separate files """
    sites_to_snotel_stations = pd.read_csv(Path(path_to_folder, 'sites_to_snotel_stations.csv'))
    station_metadata = pd.read_csv(Path(path_to_folder, 'station_metadata.csv'))

    all_files = list(path_to_folder.iterdir())
    all_files.sort()

    pass
