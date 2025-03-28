"""
Helper functions for loading both the raw, interim and processed windmill and wather datasets, 

"""

import pandas as pd
import os

import geopandas as gpd
from shapely.geometry import Polygon




##################
# Windmill data
##################

def parquet_to_df(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def parquet_to_gdf(path: str) -> gpd.GeoDataFrame:
    return gpd.read_parquet(path)

def get_raw_windmill_meta_data_df(path: str = "../data/raw/windmill/masterdatawind.parquet") -> pd.DataFrame:
    return parquet_to_df(path)
    
def get_raw_windmill_prod_data_df(base_path: str = "../data/raw/windmill/settlement/", year: int = 2018) -> pd.DataFrame:
    assert year in [2018, 2019], "production data is only available from 2018 and 2019"
    full_windmill_prod_path = os.path.join(base_path, f"{year}.parquet")    
    return parquet_to_df(full_windmill_prod_path)
    
def get_interim_windmill_data_gdf(base_path: str = "../data/interim/windmill/", year: int = 2018) -> gpd.GeoDataFrame:
    assert year in [2018, 2019]
    full_windmill_path = os.path.join(base_path, f"windmill_{year}.parquet")
    return parquet_to_gdf(full_windmill_path)

def get_interim_weather_data_gdf(base_path: str = "../data/interim/weather/", year: int = 2018) -> gpd.GeoDataFrame:
    assert year in [2018, 2019]
    full_weather_path = os.path.join(base_path, f"weather_{year}.parquet")
    return parquet_to_gdf(full_weather_path)




#################
# Weather data
#################

def json_to_df(path: str) -> pd.DataFrame:    
    return pd.read_json(path, lines=True, engine="pyarrow")


def normalize_json_df_col(df: pd.DataFrame, col: str = "properties") -> pd.DataFrame:
    df = df.join(pd.json_normalize(df[col])).drop(columns=[col])
    return df.copy()


def get_available_weather_obs_dates(base_path: str = "../data/raw/weather/climate/", 
                                    year: int = 2018) -> list[str]:
    full_dir_path = os.path.join(base_path, str(year))
    all_files = os.listdir(full_dir_path)
    
    all_dates = sorted([file_name[:10] for file_name in all_files])
    
    return all_dates
    

def get_raw_daily_weather_obs_df(base_path: str = "../data/raw/weather/climate/",
                                 date : str = "2018-01-01") -> pd.DataFrame:
    year = date[:4] #first 4 chars of date string
    
    path = os.path.join(base_path, year, f"{date}.txt")
    
    df = json_to_df(path)
    
    normalized_df = normalize_json_df_col(df)
    
    return normalized_df.copy()











def extract_weather_obs_grid_as_gdf(df: pd.DataFrame, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:   
    #only save one row per Cell ID
    df_grid = df.drop_duplicates(subset=["cellId"]).copy()
    
    #convert coordinate dict to actual spatial polygon
    df_grid["geometry"] = df_grid["geometry"].apply(lambda x: Polygon(x["coordinates"][0])) # the [0] is used as list of coordinates is nested
    
    gdf = gpd.GeoDataFrame(df_grid, geometry="geometry", crs=crs)
    
    
    return gdf[["geometry", "cellId"]]



