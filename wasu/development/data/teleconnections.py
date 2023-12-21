from pathlib import Path

import pandas as pd

NUMBER_BY_MONTH = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                   'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}

NUMBER_BY_MONTH_NAME = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}


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
