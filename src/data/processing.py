import numpy as np

import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from typing import Sequence


###########################
# General Helper Functions
###########################

def bbox_of_gdf(gdf: gpd.geodataframe.GeoDataFrame, crs: str) -> gpd.geodataframe.GeoDataFrame:
    """Given af geoPandas df (gdf) and a crs, return a new gdf containing 
    a polygon corresponding to the minimum bbox surrounding the initial gdf
    """    
    x_min, y_min, x_max, y_max =  gdf.total_bounds
    bbox_poly = box(x_min, y_min, x_max, y_max)
    
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_poly], crs=crs)
    
    return bbox_gdf


def change_df_col_dtypes(df: pd.DataFrame, dtype_dict: dict) -> pd.DataFrame:    
    df = df.copy() #in case a slice or view is passed instead of copy - prevents SettingWithCopyWarning 
    return df.astype(dtype_dict)


def change_df_cols_to_datetime(df: pd.DataFrame, col_list: list[str], utc: bool = False) -> pd.DataFrame:
    df = df.copy() #in case a slice or view is passed instead of copy - prevents SettingWithCopyWarning 
    df[col_list] = df[col_list].apply(pd.to_datetime, errors="coerce", utc=utc)
    return df


def convert_df_to_gdf(df: pd.DataFrame, geometry: str | Sequence[BaseGeometry], crs: str) -> gpd.geodataframe:
    gdf = gpd.GeoDataFrame(
        df,
        geometry=geometry,
        crs =crs
    ) 
    return gdf


def extract_subset_of_df_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].copy()


def filter_rows_by_col_values (df: pd.DataFrame | gpd.GeoDataFrame,
                               col: str,
                               values_of_interest: list[str]) -> pd.DataFrame | gpd.GeoDataFrame:
    
    mask = df[col].isin(values_of_interest)
    df = df[mask].copy()
    
    return df


def convert_CET_col_to_UTC(df: pd.DataFrame,
                           cet_col: str = "TIME_CET",
                           utc_col: str = "TIME_UTC") -> pd.DataFrame:

    df = df.copy() 
    
    df[utc_col] = pd.to_datetime(df[cet_col], utc=True) - pd.Timedelta(hours=1)

    return df
    
    
def convert_col_vals_to_new_cols(df: pd.DataFrame | gpd.GeoDataFrame, 
                                 col_to_expand: str = "parameterId",
                                 values_used_in_new_col: str = "value",
                                 cols_to_keep: list = ["geometry",
                                                       "cellId",
                                                       "from",
                                                       "to",
                                                       "timeResolution"]
                                 ) -> pd.DataFrame | gpd.GeoDataFrame:
    df = df.copy()
    
    expanded_df = df.pivot_table(index=cols_to_keep,
                                 columns=col_to_expand,
                                 values=values_used_in_new_col).reset_index()
    
    expanded_df.columns.name = None
    return expanded_df





#####################################
# Windmill-specific Helper Functions
#####################################

def change_windmill_prod_resolution(df: pd.DataFrame, new_res: str) -> pd.DataFrame:
    
    assert new_res in ["30min", "1h", "2h", "3h", "5h", "12h", "1d"], "Choose from the following resolutions: ['1h', '2h', '3h', '5h', '12h', '1d']"
    df = df.copy()
        
    #df = df.groupby([pd.Grouper(key= "TIME_CET", freq=new_res), "GSRN"], observed=True).sum().reset_index()
    df = df.groupby([pd.Grouper(key= "TIME_UTC", freq=new_res), "GSRN"], observed=True).sum().reset_index()
    
    return df


def convert_windmill_meta_df_to_gdf(df: pd.DataFrame, crs: str = "EPSG:32632") -> gpd.GeoDataFrame:
    geometry = gpd.points_from_xy(df["UTM_x"], df["UTM_y"])
    return convert_df_to_gdf(df, geometry=geometry, crs=crs)


def mask_windmills_from_id_list(df: pd.DataFrame, ids: list[str], drop_duplicate_in_col: str | None = None) -> pd.DataFrame:
    mask = df["GSRN"].isin(ids)
    df = df[mask].copy() #don't want to risk modifying OG df
    
    if drop_duplicate_in_col != None:
        df = df.drop_duplicates(subset=[drop_duplicate_in_col])
    
    return df


def add_polygon_id_to_points_within_polygon(polygons_gdf: gpd.GeoDataFrame,
                                            points_gdf: gpd.GeoDataFrame,
                                            id_col: str) -> gpd.GeoDataFrame:
    
    points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    gdf_joined = gpd.sjoin(points_gdf, polygons_gdf[[id_col, 'geometry']], how='left', predicate='within')

    points_gdf[id_col] = gdf_joined[id_col]
    
    return points_gdf


def add_wind_dir_cos_and_sin_from_wind_dir_degree(df: pd.DataFrame | gpd.GeoDataFrame,
                                                  wind_degree_col: str = "",
                                                  wind_sin_col: str = "",
                                                  wind_cos_col: str = "") -> pd.DataFrame | gpd.GeoDataFrame:
    df = df.copy()
    
    df[wind_sin_col] = np.sin(np.radians(df[wind_degree_col]))
    df[wind_cos_col] = np.cos(np.radians(df[wind_degree_col]))

    return df














