import geopandas as gpd

from tqdm import tqdm
import os

import numpy as np
import pandas as pd
from shapely.geometry import Polygon


import sys 
sys.path.append("../")

######################
# Helper functions (see ../src/data/*/* for implementation)
######################

from src.data.loading import (
    get_raw_daily_weather_obs_df,
    get_raw_windmill_meta_data_df,
    get_raw_windmill_prod_data_df,
    get_available_weather_obs_dates
)

from src.data.processing import (
    add_polygon_id_to_points_within_polygon,
    bbox_of_gdf,
    change_df_col_dtypes,
    change_df_cols_to_datetime,
    change_windmill_prod_resolution,
    convert_df_to_gdf,
    extract_subset_of_df_columns,
    mask_windmills_from_id_list,
    convert_windmill_meta_df_to_gdf,
    filter_rows_by_col_values,
    convert_col_vals_to_new_cols,
    convert_CET_col_to_UTC
)



######################
# HYPER-PARAMETERS:
######################

#CREATE SMALL VERSION OF DATA?
CREATE_SUBSET_DATA = False #if True - only store weather data for one cell AND only use first 31 days worth of data


# Paths for storing the interim data
if CREATE_SUBSET_DATA:
    INTERIM_YEARLY_WEATHER_OBS_PATH = "../data/interim/sample_dataset/weather/"
    INTERIM_COMBINED_WINDMILL_PATH = "../data/interim/sample_dataset/windmill/"
else:
    INTERIM_YEARLY_WEATHER_OBS_PATH = "../data/interim/full_dataset/weather/"
    INTERIM_COMBINED_WINDMILL_PATH = "../data/interim/full_dataset/windmill/"
    


# Only use windmills within the weather observation grid cells specified here:
if CREATE_SUBSET_DATA:  
    WEATHER_OBS_CELLIDS_OF_INTEREST = [ #could be useful to inspect map in ../notbooks/init_EDA.ipynb
        "10km_629_45",
    ]
else:
    WEATHER_OBS_CELLIDS_OF_INTEREST = [ #could be useful to inspect map in ../notbooks/init_EDA.ipynb
        "10km_629_45",
        "10km_629_46",
        "10km_628_45",
        "10km_628_46",
    ]

RELEVANT_WINDMILL_META_DATA_COLS = [
    "GSRN",             #windmill id
    "cellId",           #the polygon id of the weather obs grid cell which the windmill reside in
    #"Turbine_type",     #...
    #"Placement",        #...
    "UTM_x",            #...
    "UTM_y",            #...
    #"Capacity_kw",      #
    #"Rotor_diameter",   #
]

RELEVANT_WINDMILL_PROD_DATA_COLS = [ #Note: this really shouldn't be changed, all are needed
    "GSRN",     #for matching with meta data
    "VAERDI",   #the actual production data
    #"TIME_CET", #the time (CET timezone) of the production data
    "TIME_UTC", #the time (UTC timezone) of the production data
]


# The wanted resolution of the production data (minimum is 15min)
WINDMILL_PROD_RESOLUTION = "1h"



###################
# Helper functions 
###################

####
# Windmill 
####

def mask_windmill_data(prod_df: gpd.GeoDataFrame,
                       meta_df: pd.DataFrame | gpd.GeoDataFrame,
                       weather_df: gpd.GeoDataFrame
                       ) -> list[gpd.GeoDataFrame]:
    
    #extract the bbox of the grid of the  weather observation cells of interest
    weather_obs_grid_of_interest_gdf = weather_df.drop_duplicates(subset=["cellId"]).copy()      
    bbox_of_weather_obs_of_interest = bbox_of_gdf(weather_obs_grid_of_interest_gdf, crs="EPSG:4326")
    
    #find windmills located within the bbox of the weather observation data
    bbox_of_weather_obs_of_interest = bbox_of_weather_obs_of_interest.to_crs(meta_df.crs) #ensure the two GeoDataFrames use same crs
    windmills_within_weather_grid_gdf = meta_df[meta_df.geometry.within(bbox_of_weather_obs_of_interest.geometry.iloc[0])]

    #add weather grid id to the windmills within that grid
    windmills_within_weather_grid_gdf = add_polygon_id_to_points_within_polygon(polygons_gdf = weather_obs_grid_of_interest_gdf,
                                                        points_gdf = windmills_within_weather_grid_gdf,
                                                        id_col="cellId")
    
    # Mask windmill data based on id (GSRN) of windmills within bbox of weather observations of interest
    ids_of_windmills_within_weather_obs_bbox = list(windmills_within_weather_grid_gdf["GSRN"].unique())
    masked_windmill_meta_data_df = mask_windmills_from_id_list(df = windmills_within_weather_grid_gdf, 
                                                            ids = ids_of_windmills_within_weather_obs_bbox,
                                                            drop_duplicate_in_col="GSRN")
    masked_windmill_prod_df = mask_windmills_from_id_list(df = prod_df,
                                                    ids = ids_of_windmills_within_weather_obs_bbox)
    
    return masked_windmill_prod_df, masked_windmill_meta_data_df
        
    
def process_windmill_prod_data(df: pd.DataFrame | gpd.GeoDataFrame,
                               relevant_cols: list[str]
                               ) -> pd.DataFrame | gpd.GeoDataFrame :
    
    df = change_df_cols_to_datetime(df, ["TIME_CET"])
    df = change_df_col_dtypes(df, {"GSRN": "str", "TS_ID": "category", "VAERDI": "float"})
    df = convert_CET_col_to_UTC(df)
    df = extract_subset_of_df_columns(df, cols=relevant_cols)
    df = change_windmill_prod_resolution(df, "1h")
    
    return df


def combine_and_save_windmill_data(prod_df: pd.DataFrame | gpd.GeoDataFrame,
                                   meta_df: pd.DataFrame | gpd.GeoDataFrame,
                                   storage_path: str,
                                   file_name: str,
                                   as_gdf: bool=True,
                                   crs: str = "EPSG:32632") -> None:
    
    combined_windmill_df = pd.merge(prod_df, meta_df, on="GSRN", how="left")
    
    #create directory if it does not already exists
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    
    windmill_storage_path = os.path.join(INTERIM_COMBINED_WINDMILL_PATH, file_name)
    
    if as_gdf:
        geometry = gpd.points_from_xy(combined_windmill_df["UTM_x"], combined_windmill_df["UTM_y"])
        combined_windmill_gdf = convert_df_to_gdf(combined_windmill_df, geometry=geometry, crs=crs)
        combined_windmill_gdf.to_parquet(windmill_storage_path)
    
    else:
        combined_windmill_df.to_parquet(windmill_storage_path)
        
    return



####
# Weather 
####

def mask_weather_data(df: pd.DataFrame,
                      cellIds_of_interest: str,
                      weather_features_of_interest: list[str]) -> pd.DataFrame:
    #only keep grid cells of interest
    df_masked = filter_rows_by_col_values(df=df,
                                          col="cellId",
                                          values_of_interest=cellIds_of_interest)
    
    #only keep hourly observations
    df_masked = filter_rows_by_col_values(df=df_masked,
                                          col="timeResolution",
                                          values_of_interest=["hour"])
    
    #only keep weather features of interest
    df_masked = filter_rows_by_col_values(df=df_masked,
                                          col="parameterId",
                                          values_of_interest=weather_features_of_interest)

    return df_masked


def encode_wind_dir_as_sin_and_cos(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    assert "mean_wind_dir" in df.columns, '"mean_wind_dir" is not a column in the provided df'
    
    #Create two new columns with wind direction encoded as sine and consine
    df["mean_wind_dir_sin"] = np.sin(df["mean_wind_dir"] * (np.pi / 180))
    df["mean_wind_dir_cos"] = np.cos(df["mean_wind_dir"] * (np.pi / 180))
    
    return df.copy()


def process_weather_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
    
    #convert time related columns to datetime       
    df = change_df_cols_to_datetime(df, col_list=['calculatedAt', 'created', 'from', 'to'], utc=True)
    
    #convert the dataframe to a geodataframe
    df["geometry"] = df["geometry"].apply(lambda x: Polygon(x["coordinates"][0])) # the [0] is used as list of coordinates is nested
    weather_gdf = convert_df_to_gdf(df=df, geometry="geometry", crs="EPSG:4326")

    #convert unique values in "parameterId" to new columns with value corresponding to "value" column
    weather_gdf = convert_col_vals_to_new_cols(weather_gdf,
                                               col_to_expand = "parameterId",
                                               values_used_in_new_col = "value")

    #have to do this again... (previous function converts to reguælar df) 
    weather_gdf = convert_df_to_gdf(df=weather_gdf, geometry="geometry", crs="EPSG:4326")

    #encode wind dir
    weather_gdf = encode_wind_dir_as_sin_and_cos(weather_gdf)

    return weather_gdf



#########################
# Main / driver function
#########################

def main():
    
    if CREATE_SUBSET_DATA:
        print("CREATING SMALL SUBSET DATASET...")
    
    years = [2018, 2019] if not CREATE_SUBSET_DATA else [2018] # --> have yet to download 2019 weather data (HPC is down)   
    
    #Loaded outside for-loop for efficiency --> does not depend on year
    windmill_meta_df = get_raw_windmill_meta_data_df()
    windmill_meta_gdf = convert_windmill_meta_df_to_gdf(windmill_meta_df)
    
    for year in years:
            
        ###########
        # Process weather data
        ###########
        print(f"Processing weather data from {year}")
        all_dates = get_available_weather_obs_dates(year=year)
        all_dates = all_dates if not CREATE_SUBSET_DATA else all_dates[:31]
        
        all_weather_gdfs = []
        
        print("* Concatenating daily weather observations...")
        for date in tqdm(all_dates):
            #load daily weather observation
            weather_df = get_raw_daily_weather_obs_df(date=date)
            
            #mask weather data
            all_weather_features = weather_df["parameterId"].unique()
            masked_weather_df = mask_weather_data(weather_df,
                                                  cellIds_of_interest=WEATHER_OBS_CELLIDS_OF_INTEREST,
                                                  weather_features_of_interest=all_weather_features)   
            #process weather data
            processed_masked_weather_gdf = process_weather_data(masked_weather_df)
            all_weather_gdfs.append(processed_masked_weather_gdf)

        #combine daily weather data into yearly weather data
        combined_yearly_weather_gdf = pd.concat(all_weather_gdfs)
        
        #save yearly weather data
        print("* Saving the yearly weather observations")
        weather_file_name = f"weather_{year}.parquet" if not CREATE_SUBSET_DATA else f"weather_subset_{year}.parquet"
        weather_storage_path = os.path.join(INTERIM_YEARLY_WEATHER_OBS_PATH, weather_file_name)
        
        if not os.path.exists(INTERIM_YEARLY_WEATHER_OBS_PATH):
            os.makedirs(INTERIM_YEARLY_WEATHER_OBS_PATH)
        
        combined_yearly_weather_gdf.to_parquet(weather_storage_path)
        
        
        
        ###########
        # Process windmill data
        ###########
        print(f"Processing windmill data from {year}")
        print("* Load production data")
        windmill_prod_df = get_raw_windmill_prod_data_df(year=year)
        
        print("* Mask data")
        masked_windmill_prod_df, masked_windmill_meta_df = mask_windmill_data(prod_df = windmill_prod_df,
                                                                              meta_df = windmill_meta_gdf,
                                                                              weather_df = combined_yearly_weather_gdf)    
        
        print("* Process data")
        masked_windmill_meta_df = extract_subset_of_df_columns(masked_windmill_meta_df, 
                                                               cols=RELEVANT_WINDMILL_META_DATA_COLS)
        masked_windmill_prod_df = process_windmill_prod_data(df = masked_windmill_prod_df,
                                                             relevant_cols=RELEVANT_WINDMILL_PROD_DATA_COLS)
        
        print("* Saving the combined windmill data")
        windmill_file_name = f"windmill_{year}.parquet" if not CREATE_SUBSET_DATA else f"windmill_subset_{year}.parquet"
        combine_and_save_windmill_data(prod_df = masked_windmill_prod_df,
                                       meta_df = masked_windmill_meta_df,
                                       storage_path = INTERIM_COMBINED_WINDMILL_PATH,
                                       file_name = windmill_file_name,
                                       as_gdf=True,
                                       )
        
        
        break #Remove when 2019 data is downloaded
    
if __name__ == "__main__":
    main()