import geopandas as gpd

from tqdm import tqdm
import os

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

# Only use windmills within the weather observation grid cells specified here:
WEATHER_OBS_CELLIDS_OF_INTEREST = [ #could be useful to inspect map in ../notbooks/init_EDA.ipynb
    "10km_629_45",
    "10km_629_46",
    "10km_628_45",
    "10km_628_46",
]

INTERIM_YEARLY_WEATHER_OBS_PATH = "../data/interim/weather/"

INTERIM_COMBINED_WINDMILL_PATH = "../data/interim/windmill/"


SINGLE_WEATHER_FEATURE_OF_INTEREST = [
    "mean_wind_speed"
]

SUBSET_WEATHER_FEATURES_OF_INTEREST = [
    "mean_temp",
    "mean_pressure",
    "mean_relative_hum",
    "mean_wind_speed",
    "mean_wind_dir",
    "max_wind_speed_10min"
]



RELEVANT_WINDMILL_META_DATA_COLS = [
    "GSRN",             #windmill id
    "cellId",           #the polygon id of the weather obs grid cell which the windmill reside in
    "Turbine_type",     #...
    "Placement",        #...
    "UTM_x",            #...
    "UTM_y",            #...
    "Capacity_kw",      #
    "Rotor_diameter",   #
]

RELEVANT_WINDMILL_PROD_DATA_COLS = [ #Note: this really shouldn't be changed, all are needed
    "GSRN",     #for matching with meta data
    "VAERDI",   #the actual production data
    #"TIME_CET", #the time (CET timezone) of the production data
    "TIME_UTC", #the time (UTC timezone) of the production data
]


# The wanted resolution of the production data (minimum is 15min)
WINDMILL_PROD_RESOLUTION = "1h"

# The path where the combined windmill data for 2018 and 2019 will be stored
COMBINED_WINDMILL_BASE_PATH = "../data/interim/windmill/"


#######################
## Load the raw data
#######################
#
## Windmill data
#print("Loading raw windmill data...")
#windmill_meta_df = get_raw_windmill_meta_data_df()
#windmill_prod_2018_df = get_raw_windmill_prod_data_df(year=2018)
#windmill_prod_2019_df = get_raw_windmill_prod_data_df(year=2019)
#
## Weather data
#print("Loading raw weather observation data...")
#weather_obs_df = get_raw_daily_weather_obs_df()
#weather_obs_grid_gdf = extract_weather_obs_grid_as_gdf(weather_obs_df)
#
#
#
#####################################################################
## Masking windmills within specified weather observation grid cells
#####################################################################
#
#print("Finding windmills within specified grid cells")
## Mask the weather obs grid and find bbox of the selected grid cells
#grid_mask = weather_obs_grid_gdf["cellId"].isin(WEATHER_OBS_CELLIDS_OF_INTEREST)
#weather_obs_grid_of_interest_gdf = weather_obs_grid_gdf[grid_mask]
#bbox_of_weather_obs_of_interest = bbox_of_gdf(weather_obs_grid_of_interest_gdf, crs="EPSG:4326")
#
## Convert windmill meta data to gpd.GeoDataFrame
#geometry = gpd.points_from_xy(windmill_meta_df["UTM_x"], windmill_meta_df["UTM_y"])
#windmill_meta_gdf = convert_df_to_gdf(windmill_meta_df, geometry=geometry, crs="EPSG:32632")
#
## Find windmills within the weather observation grid cells of interest
#bbox_of_weather_obs_of_interest = bbox_of_weather_obs_of_interest.to_crs(windmill_meta_gdf.crs) #ensure the two GeoDataFrames use same crs
#windmills_within_weather_grid_gdf = windmill_meta_gdf[windmill_meta_gdf.geometry.within(bbox_of_weather_obs_of_interest.geometry.iloc[0])]
#print("Initial shape of windmill meta df:", windmills_within_weather_grid_gdf.shape)
#
## Add weather grid id to the windmills within that grid
#windmills_within_weather_grid_gdf = add_polygon_id_to_points_within_polygon(polygons_gdf = weather_obs_grid_of_interest_gdf,
#                                                           points_gdf = windmills_within_weather_grid_gdf,
#                                                           id_col="cellId")
#
## Mask windmill data based on id (GSRN) of windmills within bbox of weather observations of interest
#print("Masking the windmill meta and production datasets")
#ids_of_windmills_within_weather_obs_bbox = list(windmills_within_weather_grid_gdf["GSRN"].unique())
#masked_windmill_meta_data_df = mask_windmills_from_id_list(df = windmills_within_weather_grid_gdf, 
#                                                           ids = ids_of_windmills_within_weather_obs_bbox,
#                                                           drop_duplicate_in_col="GSRN")
#masked_windmill_prod_2018_df = mask_windmills_from_id_list(df = windmill_prod_2018_df,
#                                                  ids = ids_of_windmills_within_weather_obs_bbox)
#masked_windmill_prod_2019_df = mask_windmills_from_id_list(df = windmill_prod_2019_df,
#                                                  ids = ids_of_windmills_within_weather_obs_bbox)
#
#
#print("Shape of windmill meta data after dropping duplicate rows (GSRN column)")
#print("meta", masked_windmill_meta_data_df.shape)
#print("2018", masked_windmill_prod_2018_df.shape)
#print("2019", masked_windmill_prod_2019_df.shape)
#
#
#
#############################
## Process the windmill data
#############################
#
## Change dtypes of the windmill prod data (the meta data comes with correct dtypes)
#print("Changing dtypes of windmill production data")
#masked_windmill_prod_2018_df = change_df_cols_to_datetime(masked_windmill_prod_2018_df, ["TIME_CET"])
#masked_windmill_prod_2018_df = change_df_col_dtypes(masked_windmill_prod_2018_df, {"GSRN": "str", "TS_ID": "category", "VAERDI": "float"})
#masked_windmill_prod_2019_df = change_df_cols_to_datetime(masked_windmill_prod_2019_df, ["TIME_CET"])
#masked_windmill_prod_2019_df = change_df_col_dtypes(masked_windmill_prod_2019_df, {"GSRN": "str", "TS_ID": "category", "VAERDI": "float"})
#
## Add UTC timezone column to production data:
#masked_windmill_prod_2018_df = add_new_timezone_col_to_df(masked_windmill_prod_2018_df,
#                                                            old_col_name="TIME_CET",
#                                                            new_col_name="TIME_UTC")
#masked_windmill_prod_2019_df = add_new_timezone_col_to_df(masked_windmill_prod_2019_df,
#                                                            old_col_name="TIME_CET",
#                                                            new_col_name="TIME_UTC")
#
## Select relevant features
#print("Selecting relevant features")
#masked_windmill_meta_data_df = extract_subset_of_df_columns(masked_windmill_meta_data_df, cols=RELEVANT_WINDMILL_META_DATA_COLS)
#masked_windmill_prod_2018_df = extract_subset_of_df_columns(masked_windmill_prod_2018_df, cols=RELEVANT_WINDMILL_PROD_DATA_COLS)
#masked_windmill_prod_2019_df = extract_subset_of_df_columns(masked_windmill_prod_2019_df, cols=RELEVANT_WINDMILL_PROD_DATA_COLS)
#
## Change resolution of production data
#print("Change resolution of production data to")
#masked_windmill_prod_2018_df = change_windmill_prod_resolution(masked_windmill_prod_2018_df, "1h")
#masked_windmill_prod_2019_df = change_windmill_prod_resolution(masked_windmill_prod_2019_df, "1h")
#
#
#
##################################
## Combine windmill data and save
##################################
#
## Joining windmill datasets
#print("Joining windmill datasets")
#combined_2018_windmill_df = pd.merge(masked_windmill_prod_2018_df, masked_windmill_meta_data_df, on="GSRN", how="left")
#combined_2019_windmill_df = pd.merge(masked_windmill_prod_2019_df, masked_windmill_meta_data_df, on="GSRN", how="left")
#
## Convert the windmill data to GeoDataFrame (to ensure it got a geometry column)
#print("Converting the data to gpd.GeoDataFrame")
#geometry = gpd.points_from_xy(combined_2018_windmill_df["UTM_x"], combined_2018_windmill_df["UTM_y"])
#combined_2018_windmill_gdf = convert_df_to_gdf(combined_2018_windmill_df, geometry=geometry, crs="EPSG:32632")
#geometry = gpd.points_from_xy(combined_2019_windmill_df["UTM_x"], combined_2019_windmill_df["UTM_y"])
#combined_2019_windmill_gdf = convert_df_to_gdf(combined_2019_windmill_df, geometry=geometry, crs="EPSG:32632")
#
#print("Saving the windmill data")
#os.path.join(COMBINED_WINDMILL_BASE_PATH, "2018.parquet")
#combined_2018_windmill_gdf.to_parquet(os.path.join(COMBINED_WINDMILL_BASE_PATH, "2018.parquet"))
#combined_2019_windmill_gdf.to_parquet(os.path.join(COMBINED_WINDMILL_BASE_PATH, "2019.parquet"))








############
# Windmill 
############

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
                                   path: str,
                                   as_gdf: bool=True,
                                   crs: str = "EPSG:32632") -> None:
    
    combined_windmill_df = pd.merge(prod_df, meta_df, on="GSRN", how="left")
    
    if as_gdf:
        geometry = gpd.points_from_xy(combined_windmill_df["UTM_x"], combined_windmill_df["UTM_y"])
        combined_windmill_gdf = convert_df_to_gdf(combined_windmill_df, geometry=geometry, crs=crs)
        combined_windmill_gdf.to_parquet(path)
    
    else:
        combined_windmill_df.to_parquet(path)
        
    return



############
# Weather 
############

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

    return weather_gdf










def main():
    
    years = [2018]#, 2019]    
    
    windmill_meta_df = get_raw_windmill_meta_data_df()
    windmill_meta_gdf = convert_windmill_meta_df_to_gdf(windmill_meta_df)
    
    for year in years:
            
        ###########
        # WEATHER
        ###########
        print(f"Processing weather data from {year}")
        all_yearly_dates = get_available_weather_obs_dates(year=year)
        all_weather_gdfs = []
        
        print("* Concatenating daily weather observations...")
        for date in tqdm(all_yearly_dates):
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
        weather_storage_path = os.path.join(INTERIM_YEARLY_WEATHER_OBS_PATH, f"weather_{year}.parquet")
        combined_yearly_weather_gdf.to_parquet(weather_storage_path)
        
        
        
        ###########
        # WINDMILL
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
        windmill_storage_path = os.path.join(INTERIM_COMBINED_WINDMILL_PATH, f"windmill_{year}.parquet")
        combine_and_save_windmill_data(prod_df = masked_windmill_prod_df,
                                       meta_df = masked_windmill_meta_df,
                                       path = windmill_storage_path,
                                       as_gdf=True,
                                       )
        
        
        break
    
if __name__ == "__main__":
    main()