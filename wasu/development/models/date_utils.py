from typing import Optional

import pandas as pd


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
