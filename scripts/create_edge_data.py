import os
import pandas as pd

import utm

import sys
sys.path.append("../")

from src.data.loading import(
    get_interim_windmill_data_gdf,
    get_processed_data_df
)


CREATE_EDGE_DATA_FOR_SAMPLE_DATASET = True

#Path from where to load the windmill data
WINDMILL_DATA_BASE_PATH = "../data/interim/"

dataset_type = "full_dataset" if not CREATE_EDGE_DATA_FOR_SAMPLE_DATASET else "sample_dataset"
file_name = "windmill_2018.parquet" if not CREATE_EDGE_DATA_FOR_SAMPLE_DATASET else "windmill_subset_2018.parquet"

#Path to store the edge_feats data
EDGE_FEATS_STORAGE_PATH = f"../data/processed/{dataset_type}/" if not CREATE_EDGE_DATA_FOR_SAMPLE_DATASET else f"../data/processed/{dataset_type}/"

#Filename of the edge feature data
EDGE_FEAT_FILENAME = "edge_feats.csv"



def main():
    #load processed data and 
    wind_path = os.path.join(EDGE_FEATS_STORAGE_PATH, "all_features")
    full_df = get_processed_data_df(wind_path)
    processed_windmill_ids = full_df.columns.get_level_values(1).unique()
    
    #Load the interim data and ensure only windmills that are present in the 
    # final dataset is used and ensure that there is only one row per GSRN (aka windmill)
    wind_df = get_interim_windmill_data_gdf(base_path=WINDMILL_DATA_BASE_PATH, dataset_type=dataset_type, file_name=file_name)
    wind_df = wind_df[wind_df["GSRN"].isin(processed_windmill_ids)]
    wind_df = wind_df.drop_duplicates(subset=["GSRN", "UTM_x", "UTM_y"])
    
    #define index, cols and data to use for pd.Multiindex
    #the main index - all unique GSRN ids(header 0)
    index = list(wind_df["GSRN"].unique())

    #list of tuples for the multiindex (header 1)
    cols = []
    for id in index:
        cols.append((id, "dlat"))
        cols.append((id, "dlon"))


    #list of lists - dif between lat/lon of one GSRN to all other GSRN
    dist_data = []
    for i in range(len(index)):
        
        dist_data_i = []
        for j in range(len(index)):
                    
            if i == j:
                dist_data_i.append(0.0) #dlat
                dist_data_i.append(0.0) #dlon
                
            else:
                utm_x_i = wind_df["UTM_x"].iloc[i]
                utm_y_i = wind_df["UTM_y"].iloc[i]    
                lat_i, lon_i = utm.to_latlon(utm_x_i, utm_y_i, 32, "u")
                
                utm_x_j = wind_df["UTM_x"].iloc[j]
                utm_y_j = wind_df["UTM_y"].iloc[j]
                lat_j, lon_j = utm.to_latlon(utm_x_j, utm_y_j, 32, "u")
            
                dlat = lat_i - lat_j
                dlon = lon_i - lon_j
                dist_data_i.append(dlat)
                dist_data_i.append(dlon)
        
        dist_data.append(dist_data_i)




    #create the multiindex DataFrame
    edge_feats_df = pd.DataFrame(data=dist_data, 
                                index=index,
                                columns=pd.MultiIndex.from_tuples(cols))

    #Store the data
    edge_feats_df.to_csv(os.path.join(EDGE_FEATS_STORAGE_PATH, EDGE_FEAT_FILENAME))
          
    return



if __name__ == "__main__":
    main()
