from pathlib import Path

import pandas as pd
from loguru import logger


def collect_snotel_data_for_site(path_to_folder: Path, site_id: str, collect_only_in_basin: bool = False):
    """ Load SNOTEL data from separate files

    Collect data from all station for historical years merged with data
    """
    path_to_folder = path_to_folder.resolve()
    sites_to_snotel_stations = pd.read_csv(Path(path_to_folder, 'sites_to_snotel_stations.csv'))
    station_metadata = pd.read_csv(Path(path_to_folder, 'station_metadata.csv'))

    all_files = list(path_to_folder.iterdir())
    all_files.sort()

    # Get all atations which are related to the site
    site_snotel = sites_to_snotel_stations[sites_to_snotel_stations['site_id'] == site_id]
    if len(site_snotel) < 1:
        logger.warning(f'Cannot process SNOTEL files for site {site_id} due to there are no stations related to site')
        return None
    if collect_only_in_basin is True:
        site_snotel = site_snotel[site_snotel['in_basin'] == True]

        if len(site_snotel) < 1:
            logger.warning(f'Cannot process SNOTEL files for site {site_id} because there is no stations in the extend')
            return None

    stations_ids = list(site_snotel['stationTriplet'].unique())
    stations_ids = [station.replace(':', '_') for station in stations_ids]

    site_df = []
    for year_folder in all_files:
        if year_folder.is_file():
            # Metafiles
            continue

        station_files_per_year = list(year_folder.iterdir())
        station_files_per_year = [str(station.name).split('.csv')[0] for station in station_files_per_year]

        existing_stations_for_site = list(set(station_files_per_year).intersection(set(stations_ids)))
        if len(existing_stations_for_site) < 1:
            logger.warning(f'No SNOTEL stations for year {year_folder.name} and site {site_id}')
            continue

        # Collect data from different files
        for file in existing_stations_for_site:
            meta_info = station_metadata[station_metadata['stationTriplet'] == file.replace('_', ':')]
            df = pd.read_csv(Path(year_folder, f'{file}.csv'), parse_dates=['date'])
            df['station'] = file
            df['elevation'] = meta_info['elevation'].values[0]
            df['latitude'] = meta_info['latitude'].values[0]
            df['longitude'] = meta_info['longitude'].values[0]
            df['folder'] = year_folder.name
            site_df.append(df)

    site_df = pd.concat(site_df)
    return site_df
