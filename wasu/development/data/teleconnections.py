from pathlib import Path

import pandas as pd

NUMBER_BY_MONTH = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                   'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}


def read_soi_file_as_dataframe(path_to_soi: Path):
    with open(path_to_soi) as f:
        soi_lines = [line for line in f]

    soi_lines = soi_lines[3:84]
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


def collect_telecon_data_for_site(path_to_teleconnections: Path, site: str):
    """ Read prepared csv file for desired site """
    path_to_soi = Path(path_to_teleconnections, 'soi.txt')
    df = read_soi_file_as_dataframe(path_to_soi)

    return df.sort_values(by='YEAR')
