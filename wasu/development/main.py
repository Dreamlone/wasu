from pathlib import Path
from typing import Union, Dict

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


class WasuPipeline:
    """ Class for constructing pipelines for modelling """
    node_by_name = {}

    def __init__(self):
        self.nodes_to_execute = []

    def add_node(self, name: str, from_nodes: list, params: Union[Dict, None] = None):
        """ Adding block analysis """
        if params is None:
            node = self.node_by_name[name](from_nodes=from_nodes)
        else:
            node = self.node_by_name[name](from_nodes=from_nodes, **params)

        self.nodes_to_execute.append(node)

    def run(self):
        """ Launch constructed pipeline for analysis """
        pass
