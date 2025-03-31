import geopandas as gpd

import numpy as np

import os

import pandas as pd




import sys 
sys.path.append("../")

from src.data.loading import (
    get_interim_weather_data_gdf,
    get_interim_windmill_data_gdf
)


ONE_FEATURE = [
    "mean_wind_speed"
]

SUBSET_FEATURES = [
    "mean_wind_speed",
    "Rotor_diameter",
    "mean_relative_hum"
]

PROCESSED_ROOT_PATH = "../data/processed/"

TRAIN_VAL_TEST_SPLIT = [0.6, 0.1, 0.3]




def wrangle_df(df: pd.DataFrame | gpd.GeoDataFrame) -> pd.DataFrame | gpd.GeoDataFrame:
    
    #add new columns corresponding to the # of unique values in the GSRN columns to all other columns.
        # that is, move the GSRN column as a second header below all other columns - except TIME_UTC...
    pivot_df = df.pivot(index="TIME_UTC", columns="GSRN").reset_index()

    #work-around to also add additional columns (one for each unique val in the GSRN col) to the TIME_UTC column
    gsrn_values = df.GSRN.unique()
    time_data = {}
    for gsrn in gsrn_values:
        time_data[("TIME_UTC", gsrn)] = pivot_df["TIME_UTC"]

    time_df = pd.DataFrame(time_data)

    #drop original TIME_UTC column and add 
    pivot_df = pivot_df.drop(columns=[("TIME_UTC")], level=0)
    final_df = pd.concat([time_df, pivot_df], axis=1)
    
    
    return final_df


def remove_windmills_with_less_than_p_percent_production_data(df: pd.DataFrame | pd.DataFrame,
                                                              p: float = 0.75
                                                              ) -> pd.DataFrame | gpd.GeoDataFrame:
    
    all_ids = df.columns.get_level_values(1).unique()
    
    total_rows = len(df)
    ids_to_drop = [
        id for id in all_ids if (df["VAERDI"][id] == 0).sum() / total_rows >= p
    ]    
    
    df = df.drop(columns=df.loc[:, df.columns.get_level_values(1).isin(ids_to_drop)].columns)

    return df.copy()


def select_features(df: pd.DataFrame | gpd.GeoDataFrame,
                    selected_features: list
                    ) -> pd.DataFrame | gpd.GeoDataFrame:
    
    cols_to_always_keep = ["TIME_UTC", "VAERDI"]        
    selected_features.extend(cols_to_always_keep)
    
    selected_features = list(set(selected_features))
    
    df = df[selected_features]
    
    return df.copy()


def reorder_col_order(df: pd.DataFrame | gpd.GeoDataFrame):

    all_cols = list(df.columns.get_level_values(0).unique())

    #always keep TIME_UTC as first col and VAERDI as last col
    first, last = "TIME_UTC", "VAERDI"
    sorted_cols = sorted([x for x in all_cols if x not in {first, last}])
    custom_sorting = [first] + sorted_cols + [last]
    
    df = df.reindex(columns=custom_sorting, level=0)
    
    return df.copy()

        
def process_data(df: pd.DataFrame | gpd.GeoDataFrame,
                         selected_features: list[str],
                         p: int = 0.75) -> pd.DataFrame | gpd.GeoDataFrame:
    
    df = remove_windmills_with_less_than_p_percent_production_data(df, p)
    
    df = select_features(df, selected_features)
    
    #df = remove_redundant_columns(df, redundant_cols) #
    
    df = reorder_col_order(df)
    
    return df.copy()


def train_val_test_fraction_split(df: pd.DataFrame | gpd.GeoDataFrame,
                                  split: list[float, float, float] = [0.6, 0.1, 0.3],
                                  time_col: str = "TIME_UTC"
                                  ) -> list[pd.DataFrame | gpd.GeoDataFrame]:
    
    assert sum(split) == 1.0
    
    #Ensure that the time columns (level 0 of multiindex) is sorted. 
        #BEWARE: will sort hierarchically by the order of columns, 
        # so will likely cause problems if the times vary across the
        # different windmills --> not a problem in my current setup tho
    time_utc_cols = [col for col in df.columns if col[0] == "TIME_UTC"]
    df = df.sort_values(by=time_utc_cols)
    
    total_rows = len(df)
    train_frac = split[0]
    val_frac = split[1]
    num_of_train_cols = int(total_rows * train_frac)
    num_of_val_cols = int(np.floor(total_rows * val_frac))
    
    train_df = df.iloc[:num_of_train_cols]
    val_df = df.iloc[num_of_train_cols:(num_of_train_cols + num_of_val_cols)]
    test_df = df.iloc[(num_of_train_cols + num_of_val_cols):]
    
    
    return train_df, val_df, test_df



def main():
    # Load data
    print("* LOADING DATA")
    windmill_gdf = get_interim_windmill_data_gdf()
    weather_gdf = get_interim_weather_data_gdf()
    
    #merge datasets into a single df
    print("* MERGING DATA")
    merged_df = windmill_gdf.merge(
        weather_gdf,
        left_on=["cellId", "TIME_UTC"],
        right_on=["cellId", "to"],
        how="inner"
    )
    
    #adjust df to proper format
    print("* WRANGLING DF")
    df = wrangle_df(merged_df)
        
    #Process data depending on # of features to use
    print("* CREATING THREE DATASETS; 'one_feature', 'subset_features', 'all_features'")
    dataset_types = ["one_feature", "subset_features", "all_features"]
    for dataset_type in dataset_types:
        print("\t * ",dataset_type)
        
        if dataset_type == "one_feature":
            final_df = process_data(df, 
                              selected_features = ONE_FEATURE)
            
        elif dataset_type == "subset_features":
            final_df = process_data(df,
                              selected_features = SUBSET_FEATURES)
        
        elif dataset_type == "all_features":
            final_df = process_data(df,
                              selected_features = list(df.columns.get_level_values(0).unique()))
            
        #Split data 
        train_df, val_df, test_df = train_val_test_fraction_split(final_df, split=TRAIN_VAL_TEST_SPLIT)
        
        #store data
        train_df.to_csv(os.path.join(PROCESSED_ROOT_PATH, dataset_type, "train/wind_data.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_ROOT_PATH, dataset_type, "val/wind_data.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_ROOT_PATH, dataset_type, "test/wind_data.csv"), index=False)

                     
    return




if __name__ == "__main__":
    main()