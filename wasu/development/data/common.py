import geopandas
import pandas as pd
from geopandas import GeoDataFrame


def prepare_points_layer(spatial_dataframe: pd.DataFrame, epsg_code: str = "4326",
                         lon: str = 'longitude', lat: str = 'latitude') -> GeoDataFrame:
    geometry = geopandas.points_from_xy(spatial_dataframe[lon], spatial_dataframe[lat])
    gdf = GeoDataFrame(spatial_dataframe, crs=f"EPSG:{epsg_code}", geometry=geometry)
    return gdf
