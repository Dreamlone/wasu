from pathlib import Path

import pandas as pd


def collect_pdsi_data_for_site(path_to_pdsi: Path, site: str):
    df = pd.read_csv(Path(path_to_pdsi, f'{site}.csv'), parse_dates=['datetime'])
    return df
