import datetime
import gzip
import os
import pickle
import shutil
from pathlib import Path
from typing import Hashable, Any, Optional, Union

import geopandas
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

import warnings
warnings.filterwarnings('ignore')

FEATURES = ['PNA', 'SOI', 'SOI_STANDART', 'day_of_year', 'max_Mean_PDSI', 'max_PREC_DAILY_x', 'max_PREC_DAILY_y', 'max_TAVG_DAILY_x', 'max_TAVG_DAILY_y', 'max_TMAX_DAILY_x', 'max_TMAX_DAILY_y', 'max_TMIN_DAILY_x', 'max_TMIN_DAILY_y', 'max_WTEQ_DAILY_x', 'max_WTEQ_DAILY_y', 'mean_Mean_PDSI', 'mean_PREC_DAILY_x', 'mean_PREC_DAILY_y', 'mean_TAVG_DAILY_x', 'mean_TAVG_DAILY_y', 'mean_TMAX_DAILY_x', 'mean_TMAX_DAILY_y', 'mean_TMIN_DAILY_x', 'mean_TMIN_DAILY_y', 'mean_WTEQ_DAILY_x', 'mean_WTEQ_DAILY_y', 'min_Mean_PDSI', 'min_PREC_DAILY_x', 'min_PREC_DAILY_y', 'min_TAVG_DAILY_x', 'min_TAVG_DAILY_y', 'min_TMAX_DAILY_x', 'min_TMAX_DAILY_y', 'min_TMIN_DAILY_x', 'min_TMIN_DAILY_y', 'min_WTEQ_DAILY_x', 'min_WTEQ_DAILY_y', 'sum_Mean_PDSI', 'sum_PREC_DAILY_x', 'sum_PREC_DAILY_y', 'sum_TAVG_DAILY_x', 'sum_TAVG_DAILY_y', 'sum_TMAX_DAILY_x', 'sum_TMAX_DAILY_y', 'sum_TMIN_DAILY_x', 'sum_TMIN_DAILY_y', 'sum_WTEQ_DAILY_x', 'sum_WTEQ_DAILY_y']


NUMBER_BY_MONTH = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                   'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

NUMBER_BY_MONTH_NAME = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


def extract_values_by_extend_through_files(raster_path: Path, site_geometry):
    geometry = [mapping(site_geometry)]

    # It's important not to set crop as True because it distort output
    with rasterio.open(raster_path) as src:
        out_image, _ = mask(src, geometry, crop=False, nodata=-100.0)
        clipped_matrix = out_image[0, :, :]

    filtered_values = np.ravel(clipped_matrix)[np.ravel(clipped_matrix) > -100]
    return filtered_values


def _transform_dat_file_into_geotiff(txt_file: Path):
    product = None
    product_code = None
    data_units = None
    with open(txt_file, 'r') as fh:
        for line in fh:
            if 'Description' in line:
                product = line.split('Description: ')[-1]
            if 'Data units' in line:
                data_units = line.split('Data units: ')[-1]
            if 'Product code' in line:
                product_code = line.split('Product code: ')[-1]

    geotiff_file = Path(txt_file.parent, f'{txt_file.name.split(".")[0]}.tif')
    with rasterio.open(txt_file) as src:
        src_profile = src.profile
        src_profile.update(driver='GTiff', compress='lzw')

        with rasterio.open(geotiff_file, 'w', **src_profile) as dst:
            dst.write(src.read())

    # Remove old files
    os.remove(str(txt_file))
    os.remove(str(Path(txt_file.parent, f'{txt_file.name.split(".")[0]}.dat')))
    if product is None or product_code is None or data_units is None:
        raise ValueError(f'Can not define product name for dat file SNODAS: {txt_file}')
    return geotiff_file, product, product_code, data_units


def read_soi_file_as_dataframe(path_to_soi: Path, start_row: int, end_row: int):
    with open(path_to_soi) as f:
        soi_lines = [line for line in f]

    soi_lines = soi_lines[start_row:end_row]
    soi_lines = [l.split('\n')[0] for l in soi_lines]

    parsed_data = []
    for l in soi_lines:
        split_row = list(filter(lambda x: len(x) > 1, l.split(' ')))
        if len(split_row) == 12:
            # Some missing data
            correct_elements = split_row[:11]
            unparsed_element = split_row[11]
            correct_elements.append(unparsed_element.split('-999')[0])
            correct_elements.append(None)
            parsed_data.append(correct_elements)
        elif len(split_row) == 13:
            parsed_data.append(split_row)

    dataframe = pd.DataFrame(parsed_data[1:], columns=parsed_data[0])
    new_df = []
    for column in dataframe.columns[1:]:
        dataframe[column] = dataframe[column].astype(float)
        current_month_observations = dataframe[['YEAR', column]]
        current_month_observations['month'] = NUMBER_BY_MONTH[column]

        current_month_observations['YEAR'] = current_month_observations['YEAR'] + current_month_observations['month'].astype(str)
        current_month_observations['YEAR'] = pd.to_datetime(current_month_observations['YEAR'], format='%Y%m')
        current_month_observations = current_month_observations.rename(columns={column: 'SOI'})
        new_df.append(current_month_observations[['YEAR', 'SOI']])

    return pd.concat(new_df)


def read_pna_file(path_to_pna: Path):
    with open(path_to_pna) as f:
        pna_lines = [line for line in f]

    pna_lines = [l.split('\n')[0] for l in pna_lines]
    parsed_data = []
    for i, l in enumerate(pna_lines):
        split_row = list(filter(lambda x: len(x) > 1, l.split(' ')))
        if i == 0:
            first_row = ['YEAR']
            first_row.extend(split_row)
            parsed_data.append(first_row)
        else:
            parsed_data.append(split_row)

    dataframe = pd.DataFrame(parsed_data[1:], columns=parsed_data[0])
    new_df = []
    for column in dataframe.columns[1:]:
        dataframe[column] = dataframe[column].astype(float)
        current_month_observations = dataframe[['YEAR', column]]
        current_month_observations['month'] = NUMBER_BY_MONTH_NAME[column]

        current_month_observations['YEAR'] = current_month_observations['YEAR'] + current_month_observations['month'].astype(str)
        current_month_observations['YEAR'] = pd.to_datetime(current_month_observations['YEAR'], format='%Y%m')
        current_month_observations = current_month_observations.rename(columns={column: 'PNA'})
        new_df.append(current_month_observations[['YEAR', 'PNA']])

    return pd.concat(new_df)


def collect_telecon_data_for_site(path_to_teleconnections: Path, site: str):
    """ Read prepared csv file for desired site """
    path_to_soi = Path(path_to_teleconnections, 'soi.txt')
    path_to_pna = Path(path_to_teleconnections, 'pna.txt')

    df_pna = read_pna_file(path_to_pna)
    df_pna = df_pna.sort_values(by='YEAR')

    df = read_soi_file_as_dataframe(path_to_soi, 3, 84)
    df = df.sort_values(by='YEAR')

    df_second_part = read_soi_file_as_dataframe(path_to_soi, 87, 161)
    df_second_part = df_second_part.sort_values(by='YEAR')
    df['SOI_STANDART'] = df_second_part['SOI']
    df = df.merge(df_pna, on='YEAR')

    return df.dropna()


def generate_datetime_into_julian(dataframe: pd.DataFrame, datetime_column: str, julian_column: str,
                                  round_julian: Optional[int] = None):
    """ Transform current datetime column into julian dates """
    dataframe[julian_column] = pd.DatetimeIndex(dataframe[datetime_column]).to_julian_date()
    if round_julian is not None:
        dataframe[julian_column] = dataframe[julian_column].round(round_julian)

    return dataframe


def get_julian_date_from_datetime(current_date, round_julian: Optional[int] = None,
                                  offset_years: Optional[int] = None, offset_days: Optional[int] = None):
    """ Convert datetime object into julian """
    date_df = pd.DataFrame({'date': [current_date]})

    # Apply offset (if required)
    if offset_years is not None and offset_days is not None:
        date_df = date_df - pd.DateOffset(years=offset_years, days=offset_days)
    elif offset_years is not None:
        date_df = date_df - pd.DateOffset(years=offset_years)
    elif offset_days is not None:
        date_df = date_df - pd.DateOffset(days=offset_days)

    date_df = generate_datetime_into_julian(date_df, 'date', 'julian_datetime', round_julian)
    date_julian = date_df['julian_datetime'].values[0]

    return date_julian

def collect_snotel_data_for_site(year: int, path_to_folder: Path, site_id: str, collect_only_in_basin: bool = False):
    needed_years = [year, year - 1]
    path_to_folder = path_to_folder.resolve()
    sites_to_snotel_stations = pd.read_csv(Path(path_to_folder, 'sites_to_snotel_stations.csv'))
    station_metadata = pd.read_csv(Path(path_to_folder, 'station_metadata.csv'))

    all_files = list(path_to_folder.iterdir())
    all_files.sort()

    # Get all atations which are related to the site
    site_snotel = sites_to_snotel_stations[sites_to_snotel_stations['site_id'] == site_id]
    if collect_only_in_basin is True:
        site_snotel = site_snotel[site_snotel['in_basin'] == True]

        if len(site_snotel) < 1:
            return None

    stations_ids = list(site_snotel['stationTriplet'].unique())
    stations_ids = [station.replace(':', '_') for station in stations_ids]

    site_df = []
    for year_folder in all_files:
        if year_folder.is_file():
            # Metafiles
            continue

        current_year = str(year_folder.name).split('FY')[-1]
        if int(current_year) not in needed_years:
            continue

        station_files_per_year = list(year_folder.iterdir())
        station_files_per_year = [str(station.name).split('.csv')[0] for station in station_files_per_year]

        existing_stations_for_site = list(set(station_files_per_year).intersection(set(stations_ids)))
        if len(existing_stations_for_site) < 1:
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

def _unpack_data(archive_files: list, final_path: Path):
    ######################
    # PROCESS TEXT FILES #
    ######################
    for file in archive_files:
        if '.txt' in str(file.name):
            unpacked_text_file = Path(final_path, f'{file.name.split(".")[0]}.txt')
            with gzip.open(file, 'rb') as f_in:
                with open(unpacked_text_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    #####################
    # PROCESS DAT FILES #
    #####################
    for file in archive_files:
        if '.dat' in str(file.name):
            unpacked_dat_file = Path(final_path, f'{file.name.split(".")[0]}.dat')
            with gzip.open(file, 'rb') as f_in:
                with open(unpacked_dat_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)


def collect_snodas_data_for_site(issue_date, aggregation_days: int, path_to_folder: Path,
                                 data_dir: Path, preprocessed_dir: Path, site_id: str):
    """ Collect pandas dataframe for site """

    # Define which files are needed
    end_collection = issue_date - datetime.timedelta(days=1)
    start_collection = issue_date - datetime.timedelta(days=aggregation_days)

    paths_to_archives = []
    days_to_collect = pd.date_range(start_collection, end_collection, freq='1D')
    for day in days_to_collect:
        name_of_archive = day.strftime('%Y%m%d')
        full_name = Path(path_to_folder, f'FY{day.year}', f'SNODAS_{name_of_archive}.tar').resolve()
        if full_name.is_file() is True:
            paths_to_archives.append(full_name)

    #################
    # Unpack folder #
    #################
    folder_to_unpack_files = Path(preprocessed_dir, 'unpacked_snodas_archives').resolve()
    if len(paths_to_archives) > 3:
        # To speed up calculations
        paths_to_archives = paths_to_archives[:2]

    metadata = []
    for archive in paths_to_archives:
        archive_base_name = archive.name.split('.')[0]
        date_info = archive_base_name.split('_')[-1]
        date_info = datetime.datetime.strptime(date_info, '%Y%m%d')

        path_to_extract = Path(folder_to_unpack_files, archive_base_name)
        path_to_extract.mkdir(exist_ok=True, parents=True)

        shutil.unpack_archive(filename=archive, extract_dir=path_to_extract)
        final_path = Path(folder_to_unpack_files, f"{archive.name.split('.')[0]}_processed")
        final_path.mkdir(exist_ok=True, parents=True)

        archive_files = list(path_to_extract.iterdir())
        _unpack_data(archive_files, final_path)
        shutil.rmtree(path_to_extract)

        ####################################
        # Transform dat files into geotiff #
        ####################################
        txt_files = list(filter(lambda x: '.txt' in str(x), list(final_path.iterdir())))
        for file in txt_files:
            try:
                # Transform into geotiff and save files
                geotiff_file, product, product_code, data_units = _transform_dat_file_into_geotiff(file)
                product = product.replace('\n', '')
                product_code = product_code.replace('\n', '')
                data_units = data_units.replace('\n', '')

                metadata.append(pd.DataFrame({'archive_name': [archive.name], 'datetime': [date_info],
                                              'product': [product],
                                              'product_code': [product_code], 'data_units': [data_units],
                                              'geotiff': [geotiff_file]}))
            except Exception as ex:
                pass

    if len(metadata) < 1:
        raise ValueError(f'Can not collect data from archives {len(paths_to_archives)}')
    info_df = pd.concat(metadata)
    info_df['datetime'] = pd.to_datetime(info_df['datetime'])

    ###########################################
    # COLLECT DATA FROM CREATED GEOTIFF FILES #
    ###########################################
    spatial_objects = geopandas.read_file(Path(data_dir, 'geospatial.gpkg'))
    site_geom = spatial_objects[spatial_objects['site_id'] == site_id]
    site_geometry = site_geom.geometry.values[0]

    all_info_per_site = []
    for date_info in list(info_df['datetime'].unique()):
        try:
            date_df = info_df[info_df['datetime'] == date_info]

            datetime_site_df = pd.DataFrame({'datetime': [date_info]})
            for row_id, row in date_df.iterrows():
                geotiff_file = row.geotiff
                product_name = row['product']
                # Product melt rate
                if product_name == 'Modeled melt rate, bottom of snow layers, 24-hour total':
                    product_name = 'Modeled melt rate, bottom of snow layers'

                # Snow accumulation
                if product_name == 'Scaled Snow accumulation, 24-hour total':
                    product_name = 'Snow accumulation, 24-hour total'
                if product_name == 'Scaled Snow accumulation 3 hour forecast, 24-hour total':
                    product_name = 'Snow accumulation, 24-hour total'

                # Non snow accumulation
                if product_name == 'Scaled Non-snow accumulation, 24-hour total':
                    product_name = 'Non-snow accumulation, 24-hour total'
                if product_name == 'Scaled Non-snow accumulation 3 hour forecast, 24-hour total':
                    product_name = 'Non-snow accumulation, 24-hour total'

                vals = extract_values_by_extend_through_files(geotiff_file, site_geometry)
                # Calculate statistics for current datetime
                datetime_site_df[f'mean_{product_name}'] = np.nanmean(vals)
                datetime_site_df[f'sum_{product_name}'] = np.nansum(vals)
                datetime_site_df[f'std_{product_name}'] = np.nanstd(vals)

            all_info_per_site.append(datetime_site_df)
        except Exception as ex:
            pass

    all_info_per_site = pd.concat(all_info_per_site)
    shutil.rmtree(folder_to_unpack_files)
    return all_info_per_site


def _aggregate_features_snodas(agg_snodas: pd.DataFrame):
    features_columns = ['mean_Snow accumulation, 24-hour total',
                        'mean_Non-snow accumulation, 24-hour total',
                        'mean_Modeled snow water equivalent, total of snow layers',
                        'mean_Modeled snow layer thickness, total of snow layers']
    agg_snodas = agg_snodas[features_columns]
    agg_snodas = agg_snodas.dropna()

    dataset = pd.DataFrame()
    for feature in features_columns:
        mean_value = agg_snodas[feature].mean()
        min_value = agg_snodas[feature].min()
        max_value = agg_snodas[feature].max()

        dataset[f'mean_{feature}'] = [mean_value]
        dataset[f'min_{feature}'] = [min_value]
        dataset[f'max_{feature}'] = [max_value]

    return dataset


def _aggregate_features_pdsi(agg_pdsi: pd.DataFrame):
    dataset = pd.DataFrame()
    for feature in ['Mean_PDSI']:
        mean_value = agg_pdsi[feature].mean()
        sum_value = agg_pdsi[feature].sum()
        min_value = agg_pdsi[feature].min()
        max_value = agg_pdsi[feature].max()

        dataset[f'mean_{feature}'] = [mean_value]
        dataset[f'sum_{feature}'] = [sum_value]
        dataset[f'min_{feature}'] = [min_value]
        dataset[f'max_{feature}'] = [max_value]

    return dataset


def _aggregate_features_snotel(agg_snotel: pd.DataFrame):
    agg_snotel = agg_snotel[['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY', 'WTEQ_DAILY']]
    agg_snotel = agg_snotel.dropna()

    dataset = pd.DataFrame()
    for feature in ['PREC_DAILY', 'TAVG_DAILY', 'TMAX_DAILY', 'TMIN_DAILY', 'WTEQ_DAILY']:
        mean_value = agg_snotel[feature].mean()
        sum_value = agg_snotel[feature].sum()
        min_value = agg_snotel[feature].min()
        max_value = agg_snotel[feature].max()

        dataset[f'mean_{feature}'] = [mean_value]
        dataset[f'sum_{feature}'] = [sum_value]
        dataset[f'min_{feature}'] = [min_value]
        dataset[f'max_{feature}'] = [max_value]

    return dataset


def _aggregate_features_soi(agg_soi: pd.DataFrame):
    return agg_soi.head(1)[['SOI', 'SOI_STANDART', 'PNA']]


def aggregate_data_for_issue_date(issue_date, day_of_year, dataframe,
                                  aggregation_days, label: str) -> Union[pd.DataFrame, None]:
    aggregation_end_julian = get_julian_date_from_datetime(current_date=issue_date)
    aggregation_start_julian = get_julian_date_from_datetime(current_date=issue_date,
                                                             offset_days=aggregation_days)

    aggregated = dataframe[dataframe['julian_datetime'] >= aggregation_start_julian]
    aggregated = aggregated[aggregated['julian_datetime'] < aggregation_end_julian]
    if len(aggregated) < 1:
        return None

    if label == 'snodas':
        dataset = _aggregate_features_snodas(aggregated)
    elif label == 'snotel':
        dataset = _aggregate_features_snotel(aggregated)
    elif label == 'soi':
        dataset = _aggregate_features_soi(aggregated)
    elif label == 'pdsi':
        dataset = _aggregate_features_pdsi(aggregated)
    else:
        raise ValueError(f'Not supported label {label}')

    dataset['issue_date'] = issue_date
    dataset['day_of_year'] = day_of_year
    return dataset


def collect_features_for_prediction(site_id: str, data_dir: Path, preprocessed_dir: Path, issue_date: str):
    aggregation_days_snotel = 90
    aggregation_days_pdsi = 124
    telecon_offset = 150

    path_to_snotel = Path(data_dir, 'snotel').resolve()
    path_to_teleconnections = Path(data_dir, 'teleconnections').resolve()
    path_to_pdsi = Path(preprocessed_dir, 'pdsi_csv').resolve()

    issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')

    # PREPARE PDSI
    pdsi_df = pd.read_csv(Path(path_to_pdsi, f'{site_id}.csv'), parse_dates=['datetime'])
    pdsi_df = generate_datetime_into_julian(dataframe=pdsi_df, datetime_column='datetime',
                                            julian_column='julian_datetime', round_julian=3)
    pdsi_features = aggregate_data_for_issue_date(issue_date, int(issue_date.strftime('%j')), pdsi_df,
                                                  aggregation_days_pdsi,'pdsi')

    # PREPARE TELECON
    telecon_df = collect_telecon_data_for_site(path_to_teleconnections, site_id)
    telecon_df = generate_datetime_into_julian(dataframe=telecon_df, datetime_column='YEAR',
                                               julian_column='julian_datetime', round_julian=3)
    telecon_features = aggregate_data_for_issue_date(issue_date, int(issue_date.strftime('%j')), telecon_df,
                                                     telecon_offset, 'soi')

    # PREPARE SNOTEL
    snotel_df = collect_snotel_data_for_site(issue_date.year, path_to_snotel, site_id)
    snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                              julian_column='julian_datetime', round_julian=3)
    snotel_features = aggregate_data_for_issue_date(issue_date, int(issue_date.strftime('%j')), snotel_df,
                                                    aggregation_days_snotel, 'snotel')

    # PREPARE SNOTEL 2
    snotel_df = collect_snotel_data_for_site(issue_date.year, path_to_snotel, site_id)
    snotel_df = generate_datetime_into_julian(dataframe=snotel_df, datetime_column='date',
                                              julian_column='julian_datetime', round_julian=3)
    snotel_features_short = aggregate_data_for_issue_date(issue_date, int(issue_date.strftime('%j')), snotel_df,
                                                           4, 'snotel')
    current_dataset = None
    for df in [snotel_features_short, snotel_features, telecon_features, pdsi_features]:
        if current_dataset is None:
            # First iteration
            current_dataset = df
        else:
            current_dataset = current_dataset.merge(df.drop(columns=['day_of_year']), on='issue_date')

    return np.array(current_dataset[FEATURES])


def predict(site_id: str, issue_date: str, assets: dict[Hashable, Any],
            src_dir: Path, data_dir: Path, preprocessed_dir: Path,) -> tuple[float, float, float]:
    """A function that generates a forecast for a single site on a single issue
    date. This function will be called for each site and each issue date in the
    test set.

    Args:
        site_id (str): the ID of the site being forecasted.
        issue_date (str): the issue date of the site being forecasted in
            'YYYY-MM-DD' format.
        assets (dict[Hashable, Any]): a dictionary of any assets that you may
            have loaded in the 'preprocess' function. See next section.
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.
    Returns:
        tuple[float, float, float]: forecasted values for the seasonal water supply.
        The three values should be (0.10 quantile, 0.50 quantile, 0.90 quantile).
    """
    # Prepare features for predict (see)
    try:
        features_for_predict = collect_features_for_prediction(site_id, data_dir, preprocessed_dir, issue_date)
    except Exception as ex:
        try:
            # One week
            issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')
            issue_date = issue_date - datetime.timedelta(days=7)
            issue_date = issue_date.strftime('%Y-%m-%d')
            features_for_predict = collect_features_for_prediction(site_id, data_dir, preprocessed_dir, issue_date)
        except Exception as ex:
            try:
                # One year
                issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')
                issue_date = issue_date - datetime.timedelta(days=365)
                issue_date = issue_date.strftime('%Y-%m-%d')
                features_for_predict = collect_features_for_prediction(site_id, data_dir, preprocessed_dir, issue_date)
            except Exception as ex:
                try:
                    # Two years
                    issue_date = datetime.datetime.strptime(issue_date, '%Y-%m-%d')
                    issue_date = issue_date - datetime.timedelta(days=365*2)
                    issue_date = issue_date.strftime('%Y-%m-%d')
                    features_for_predict = collect_features_for_prediction(site_id, data_dir, preprocessed_dir, issue_date)
                except Exception as ex:
                    return 200.0, 400.0, 600.0

    path_to_models = Path(src_dir, 'models').resolve()

    model_predict = {}
    for alpha, column_name in zip([0.1, 0.5, 0.9], ['volume_10', 'volume_50', 'volume_90']):
        scaler_name = f'scaler_{site_id}_{str(alpha).replace(".", "_")}.pkl'
        scaler_path = Path(path_to_models, scaler_name)

        with open(scaler_path, 'rb') as pkl:
            scaler = pickle.load(pkl)

        file_name = f'model_{site_id}_{str(alpha).replace(".", "_")}.pkl'
        model_path = Path(path_to_models, file_name)

        with open(model_path, 'rb') as pkl:
            model = pickle.load(pkl)

        predicted = model.predict(scaler.transform(features_for_predict))
        model_predict.update({column_name: float(predicted)})

    #############################################
    # Final adjustment (if values are not good) #
    #############################################
    current_min = model_predict['volume_10']
    current_max = model_predict['volume_90']
    current_mean = model_predict['volume_50']
    if current_min > current_max:
        model_predict['volume_10'] = current_max
        model_predict['volume_90'] = current_min
    if current_mean <= model_predict['volume_10'] or current_mean >= model_predict['volume_90']:
        model_predict['volume_50'] = (model_predict['volume_10'] + model_predict['volume_90']) / 2
    #######################
    # FINISH CALCULATIONS #
    #######################
    return float(model_predict['volume_10']), float(model_predict['volume_50']), float(model_predict['volume_90'])


def extract_data_from_netcdf_file(geometry, netcdf_file) -> pd.DataFrame:
    netcdf_file_name = netcdf_file.name.split('_')
    first_date = netcdf_file_name[1]
    second_date = netcdf_file_name[2].split('.nc')[0]
    datetime_labels = pd.date_range(first_date, second_date, freq='5D')

    dataframe = []
    with rasterio.open(netcdf_file) as src:
        out_image, _ = mask(src, geometry, crop=False, nodata=32767.0)
        for time_id in range(len(out_image)):
            clipped_matrix = out_image[time_id, :, :]
            filtered_values = np.ravel(clipped_matrix)[np.ravel(clipped_matrix) < 32766]

            dataframe.append([np.nanmean(filtered_values), np.nansum(filtered_values), np.nanstd(filtered_values)])

    dataframe = pd.DataFrame(dataframe, columns=['Mean_PDSI', 'Sum_PDSI', 'std_PDSI'])
    dataframe['datetime'] = datetime_labels

    return dataframe


def prepare_pdsi_data_as_table(data_dir: Path, preprocessed_dir: Path):
    spatial_objects = geopandas.read_file(Path(data_dir, 'geospatial.gpkg'))

    pdsi_path = Path(data_dir, 'pdsi').resolve()
    pdsi_results = Path(preprocessed_dir, 'pdsi_csv').resolve()
    pdsi_results.mkdir(parents=True, exist_ok=True)

    years = list(pdsi_path.iterdir())
    years.sort()
    for row_id, spatial_object in spatial_objects.iterrows():
        site_id = spatial_object.site_id
        site_geom = spatial_objects[spatial_objects['site_id'] == site_id]
        site_geometry = site_geom.geometry.values[0]
        geometry = [mapping(site_geometry)]

        dataframe_for_site = []
        for folder in years:
            files_in_folder = list(folder.iterdir())
            if len(files_in_folder) < 1:
                continue

            for netcdf_file in files_in_folder:
                df = extract_data_from_netcdf_file(geometry, netcdf_file)
                dataframe_for_site.append(df)

        dataframe_for_site = pd.concat(dataframe_for_site)
        dataframe_for_site['datetime'] = pd.to_datetime(dataframe_for_site['datetime'])
        dataframe_for_site = dataframe_for_site.sort_values(by='datetime')

        dataframe_for_site.to_csv(Path(pdsi_results, f'{site_id}.csv'), index=False)


def preprocess(src_dir: Path, data_dir: Path, preprocessed_dir: Path) -> dict[Hashable, Any]:
    """An optional function that performs setup or processing.

    Args:
        src_dir (Path): path to the directory that your submission ZIP archive
            contents are unzipped to.
        data_dir (Path): path to the mounted data drive.
        preprocessed_dir (Path): path to a directory where you can save any
            intermediate outputs for later use.

    Returns:
        (dict[Hashable, Any]): a dictionary containing any assets you want to
            hold in memory that will be passed to your 'predict' function as
            the keyword argument 'assets'.
    """
    # First there is a need to prepare three data sources: PDSI, SNODAS, SNOTEL
    prepare_pdsi_data_as_table(data_dir, preprocessed_dir)
    # folder_to_unpack_files = Path(preprocessed_dir, 'unpacked_snodas_archives').resolve()
    # folder_to_unpack_files.mkdir(exist_ok=True, parents=True)
    return {}


if __name__ == '__main__':
    print(predict('hungry_horse_reservoir_inflow', '2023-05-08', 1,
            Path('.').resolve(), Path('../data').resolve(), Path('../data').resolve()))
