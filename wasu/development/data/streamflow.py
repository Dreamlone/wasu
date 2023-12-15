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
            pass

    if len(site_df) < 1:
        return None

    site_df = pd.concat(site_df)
    site_df = site_df.sort_values(by='datetime')
    return site_df
