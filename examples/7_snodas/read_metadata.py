import datetime
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')


def read_metadata():
    """
    This script help update metadata file with paths to geotiff files
    (and aggregate names of variables to real values)
    """

    folder_to_unpack_files = Path('../../data/snodas_unpacked').resolve()
    df = pd.read_csv(Path(folder_to_unpack_files, 'metadata_old.csv'))

    print(f'Lenght: {len(df["product"].unique())}')
    print(df['product'].unique())
    updated_dataframe = []
    for year in list(folder_to_unpack_files.iterdir()):
        if year.is_file():
            # Skip metadata
            continue

        for date_in_year in list(year.iterdir()):
            date_info = date_in_year.name.split('_')[1]
            date_info = datetime.datetime.strptime(date_info, '%Y%m%d')

            # Convert into "2004-03-27"
            date_as_string = date_info.strftime('%Y-%m-%d')
            date_df = df[df['datetime'] == date_as_string]
            if len(date_df) < 1:
                continue

            updated_row = []
            for geotiff_file in list(date_in_year.iterdir()):
                if '34tS' in geotiff_file.name:
                    # 1 - SWE
                    name = 'Modeled snow water equivalent, total of snow layers'
                    product_df = date_df[date_df['product'] == name]

                elif '36tS' in geotiff_file.name:
                    # 2 - Snow thickness
                    name = 'Modeled snow layer thickness, total of snow layers'
                    product_df = date_df[date_df['product'] == name]

                elif '38wS' in geotiff_file.name:
                    # 3 - temperature
                    name = 'Modeled average temperature, SWE-weighted average of snow layers, 24-hour average'
                    product_df = date_df[date_df['product'] == name]

                elif '44bS' in geotiff_file.name:
                    # 4 -
                    name = 'Modeled melt rate, bottom of snow layers, 24-hour total'
                    product_df = date_df[date_df['product'] == name]

                    if len(product_df) < 1:
                        # 5
                        name = 'Modeled melt rate, bottom of snow layers'
                        product_df = date_df[date_df['product'] == name]

                elif '50lL' in geotiff_file.name:
                    # 6
                    name = 'Modeled snowpack sublimation rate, 24-hour total'
                    product_df = date_df[date_df['product'] == name]

                elif '39lL' in geotiff_file.name:
                    # 7
                    name = 'Modeled blowing snow sublimation rate, 24-hour total'
                    product_df = date_df[date_df['product'] == name]

                elif '25SlL01' in geotiff_file.name:
                    # 8
                    name = 'Scaled Snow accumulation, 24-hour total'
                    product_df = date_df[date_df['product'] == name]
                    if len(product_df) < 1:
                        # 9
                        name = 'Snow accumulation, 24-hour total'
                        product_df = date_df[date_df['product'] == name]
                    if len(product_df) < 1:
                        # Still not recognized - 10
                        name = 'Scaled Snow accumulation 3 hour forecast, 24-hour total'
                        product_df = date_df[date_df['product'] == name]

                elif '25SlL00' in geotiff_file.name:
                    # 11
                    name = 'Scaled Non-snow accumulation, 24-hour total'
                    product_df = date_df[date_df['product'] == name]
                    if len(product_df) < 1:
                        # 12
                        name = 'Non-snow accumulation, 24-hour total'
                        product_df = date_df[date_df['product'] == name]
                    if len(product_df) < 1:
                        # Still not recognized - 13
                        name = 'Scaled Non-snow accumulation 3 hour forecast, 24-hour total'
                        product_df = date_df[date_df['product'] == name]
                else:
                    raise ValueError(f'Unrecognized file: {geotiff_file.name}')

                if len(product_df) < 1:
                    raise ValueError(f'Unrecognized data format for file: {geotiff_file.name}')
                product_df['geotiff'] = geotiff_file
                updated_row.append(product_df)

            # Finished processing date of the year
            updated_row = pd.concat(updated_row)
            updated_dataframe.append(updated_row)

    updated_dataframe = pd.concat(updated_dataframe)
    path_to_save = Path(folder_to_unpack_files, 'metadata.csv')
    updated_dataframe.to_csv(path_to_save, index=False)


def check_possible_products():
    folder_to_unpack_files = Path('../../data/snodas_unpacked').resolve()
    df = pd.read_csv(Path(folder_to_unpack_files, 'metadata.csv'), parse_dates=['datetime'])
    df = df.sort_values(by='datetime')
    df_datetime = df[['datetime']].drop_duplicates()

    r = pd.date_range(start=min(df['datetime']), end=max(df['datetime']), freq='1d')
    df['value'] = np.arange(0, len(df))

    print(f'Lenght: {len(df["product"].unique())}')
    print(df[['product', 'data_units']].drop_duplicates())

    print(f'Missing datetime indices: {len(r) - len(df_datetime)}')
    missed_parts = list(set(list(r.unique())) - set(list(df_datetime['datetime'].unique())))
    missed_parts.sort()
    for i in missed_parts:
        print(i)


if __name__ == '__main__':
    folder_to_unpack_files = Path('../../data/snodas_unpacked').resolve()
    df = pd.read_csv(Path(folder_to_unpack_files, 'metadata.csv'), parse_dates=['datetime'])
    df = df[['archive_name', 'datetime', 'product', 'product_code', 'data_units', 'geotiff']]
    df['datetime'] = pd.to_datetime(df['datetime'], format='ISO8601').dt.strftime('%Y-%m-%d')

    path_to_save = Path(folder_to_unpack_files, 'metadata_new.csv')
    df.to_csv(path_to_save, index=False)
