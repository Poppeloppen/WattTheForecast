import os
import pandas as pd

import utm

import sys
sys.path.append("../")

from src.data.loading import(
    get_processed_data_df,
    get_raw_windmill_meta_data_df
)

CREATE_NODE_DATA_FOR_SAMPLE_DATASET = True

#Path to store the node_info data
NODE_FEATS_STORAGE_PATH = "../data/processed/full_dataset/" if not CREATE_NODE_DATA_FOR_SAMPLE_DATASET else "../data/processed/sample_dataset/"

#Filename of the edge feature data
NODE_FEAT_FILENAME = "node_info.csv"



def main():
    #load data and ensure only one row per GSRN (aka windmill)
    wind_path = os.path.join(NODE_FEATS_STORAGE_PATH, "all_features")
    full_df = get_processed_data_df(wind_path)
    processed_windmill_ids = full_df.columns.get_level_values(1).unique()
    
    #load the raw windmill meta data
    raw_windmill_meta_df = get_raw_windmill_meta_data_df()
    #select only windmills that are in the final processed data
    subset_raw_windmill_df = raw_windmill_meta_df[raw_windmill_meta_df["GSRN"].isin(processed_windmill_ids)]
    #ensure only one row per windmill ID
    subset_raw_windmill_df = subset_raw_windmill_df.drop_duplicates(subset=["GSRN", "UTM_x", "UTM_y"])
    
    #Convert UTM to lat/lon
    utm_x = subset_raw_windmill_df["UTM_x"].values
    utm_y = subset_raw_windmill_df["UTM_y"].values
    lat, lon = utm.to_latlon(utm_x, utm_y, 32, "u")
    subset_raw_windmill_df["lat"] = lat
    subset_raw_windmill_df["lon"] = lon
    
    
    #only store relevant columns
    node_feature_df = subset_raw_windmill_df[["GSRN", "lat", "lon"]]
    
        
    #store data
    node_feature_df.to_csv(os.path.join(NODE_FEATS_STORAGE_PATH, NODE_FEAT_FILENAME), index=False)
          
    return



if __name__ == "__main__":
    main()
