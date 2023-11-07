"""Code for downloading the Pacific Decadal Oscillation (PDO) Index. Use the CLI to download, for
example:

    python -m wsfr_download pdo

or use the `bulk` command to download many sources at once based on a config file.

    python -m wsfr_download bulk data_download/hindcast_test_config.yml

You can also import this module and use it as a library.

See the challenge website for more about this approved data source:
https://www.drivendata.org/competitions/254/reclamation-water-supply-forecast-dev/page/801/#pacific-decadal-oscillation-pdo-index
"""

from loguru import logger
import requests

from wsfr_download.config import DATA_ROOT

SOURCE_URL = "https://www.ncei.noaa.gov/pub/data/cmb/ersst/v5/index/ersst.v5.pdo.dat"
FILE_PATH_PARTS = ("teleconnections", "pdo.txt")


def download_pdo():
    """Download Pacific Decadal Oscillation (PDO) Index data."""
    logger.info("Downloading PDO Index data...")
    response = requests.get(SOURCE_URL)
    out_file = DATA_ROOT.joinpath(*FILE_PATH_PARTS)
    out_file.parent.mkdir(exist_ok=True, parents=True)
    with out_file.open("w") as fp:
        fp.write(response.text)
    logger.success(f"PDO Index data written to {out_file}")
