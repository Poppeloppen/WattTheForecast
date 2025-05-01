
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset
from typing import Any

class Dataset_wind_data(Dataset):
    def __init__(self, root_path: str,              #path to root dir of clean data
                 dataset_size: str,
                 dataset_features: str,             #["one_feature", "subset_features", "all_features"]
                 flag: str = "train",               #["train", "val", "test"] - I think?
                 size : list[int, int, int] = None, #[seq_len, label_len, pred_len]
                 features: str = "S",               #S: univariate, M: multivariate
                 file_name: str = "wind_data.csv",  #name of file to read
                 target: str = None,                #name of station/windmill to target
                 scale: bool = True,                #
                 timeenc: str = 0,                  #
                 freq: str = "1h",                  #
                 all_stations: bool = False,        #
                 data_step: int = 5,                #use every data_step'th point (1 for all data)
                 **_: Any                           #allow for the function to take additional arguments - won't be used
                 ) -> None:      
        
        assert flag in ["train", "test", "val"]
        flag_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = flag_map[flag]
        self.flag = flag
       
        #(notation used in paper)
        self.seq_len = size[0]      # S (the look-back window)
        self.label_len = size[1]    # L (#)
        self.pred_len = size[2]     # P (# of time-steps to predict into the future)
        
        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len      
        
        self.all_stations = all_stations
        self.data_step = data_step
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.dataset_size = dataset_size
        self.dataset_features = dataset_features
        self.file_name = file_name
        
        self.__read_data__()
        
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.dataset_size, self.dataset_features, self.flag, self.file_name)        
        df_raw = pd.read_csv(_path, header=[0,1])
        
        
        # dict with indices of windmills - {GSRN: idx}
        self._stations = {s: i for i, s in enumerate(df_raw.columns.get_level_values(1).unique())}
            
        if self.features == "M" or self.features == "MS":
            #Mask out time (keep all other features)
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != "TIME_UTC"]
            df_data = df_raw[cols_data]
            #Ensure that VAERDI (production) is the last column
            assert (df_raw.columns.get_level_values(0).unique()[-1:] == ["VAERDI"]).all()
        elif self.features == "S":
            #Only keep production data
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) == "VAERDI"]
            df_data = df_raw[cols_data]
        

        if self.scale:
            #below line is same as df_data.columns.get_level_values(0).unique()
            self.cols_meas = df_data.stack(future_stack=True).columns
            
            #fit scaler based only on training data + only keep relevant cols (as per above if-else-statement)
            if self.flag != "train":
                train_data = pd.read_csv(_path.replace(self.flag, "train"), header=[0,1])
                train_data = train_data[cols_data]
                self.scaler.fit(train_data.stack(future_stack=True).values)
                del train_data #free up memory -> train_data is no longer used
            else:
                self.scaler.fit(df_data.stack(future_stack=True).values)
            
            #from (2621, 1472) --> (2621, 23, 64), that is (#rows, #features, #GSRN). note that 23*64 = 1472
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
            data = np.stack([self.scaler.transform(data[..., i]) for i in range(data.shape[-1])], -1) #apply the scaler
        else:
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
                
                
        if not self.all_stations:
            data = data[..., self._stations[self.target]]
            data = np.expand_dims(data, -1)   
       
       
        #############
        #A BUNCH OF MISSINGNESS-HANDLING HERE... Need to check this again...
        #############
        nan_indxs = [np.where(np.isnan(data[..., i]).any(axis=1))[0] for i in range(data.shape[-1])] #list of arrays (one array for each windmill) --> each array represents the row idx with ANY missing value (across all feature cols)
        nan_indxs = [np.unique(np.concatenate([np.array([0]), nan_indxs[i], np.array([data[..., 0].shape[0] - 1])])) #create new array: [0, <..the idx of all rows containing any missing value (from above)..>, # of rows -1] --> basically just adds first and last value to NaN list
                     for i in range(len(nan_indxs))]  
        #nan_indxs is a list of arrays with # of arrays = # of windmills - each array contains the idx's of rows with at least a single NaN value      
        #eg ([0, 2620], ... * # of windmills (64))

        # Find the slices which result in valid sequences without NaNs
        valid_slices = [np.where((nan_indxs[i][1:] - nan_indxs[i][:-1] - 1) >= self.total_seq_len)[0]
                        for i in range(len(nan_indxs))]
        valid_slices = [np.vstack([nan_indxs[i][valid_slices[i]] + 1, nan_indxs[i][valid_slices[i] + 1] - 1]).T
                        for i in range(len(nan_indxs))]
        #valid_slices is once again a list of arrays with # of arrays = # of windmills - now each array contains start and end idx's of rows with NO NaN values
        #eg ([1, 2619], ... * # of windmills (64))

        # Now, construct an array which contains the valid start indices for the different sequences
        data_indxs = []
        #for each windmill
        for i in range(len(valid_slices)):
            #create array of len: (# of rows - total_seq_len + 1) filled with False - ([False, ... * (# of rows - total_seq_len + 1)])
            ###### WHY SUBTRACT total_seq_len ???? 
            start_indxs = np.zeros(data.shape[0] - self.total_seq_len + 1, dtype='bool')
            
            #get start and end idx of valid slices
            for s, e in valid_slices[i]:
                #create a range from (start idx) to (end idx - total_seq_len)
                indxs_i = np.arange(s, e - self.total_seq_len + 2, 1)
                #turn False value of all valid rows into True
                start_indxs[indxs_i] = True
            data_indxs.append(start_indxs)
        #turn list of arrays to 2d array - [array(False, True, ... True, False), ... * # of windmills] --> (2552, # of windmills)
        self.data_indxs = np.stack(data_indxs, -1)
        #eg [[False False False ... False False False]
        #   [ True  True  True ...  True  True  True]
        #   ...
        #   [ True  True  True ...  True  True  True]
        #   [ True  True  True ...  True  True  True]
        #   [False False False ... False False False]]
        #   shape: (2552, 64), where 2552 = 2621 - 70(total_seq_len) + 1
        ######################################
        
                
        #construct time array
        assert df_raw[["TIME_UTC"]].all(0).all()
        df_stamp = df_raw[["TIME_UTC"]].iloc[:, :1]
        df_stamp.columns = ["time"]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)

        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.time.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.time.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.time.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.time.apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp.time.apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(["time"], axis=1).values
        elif self.timeenc == 1:
            raise NotImplementedError("Yet to be implemented")
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        
        self.data_x = data              #the full dataset (except time)
        self.data_stamp = data_stamp    #the time dataset
        
        self.valid_indxs = np.where(self.data_indxs.any(-1))[0] #get the row idx of all rows with no missing values
        self.valid_indxs = self.valid_indxs[::self.data_step] #apply the data_step size -> only look at every data_step'th datapoint
        self.full_indx_row, self.full_indx_col = np.where(self.data_indxs[self.valid_indxs, :])
        #full_indx_row: [0, ... * # of windmills, 1, ... * # of windmills, ..., len(valid_indxs)-1 * # of windmills ]
        #full_indx_row: [0, 1, 2, ... * # of windmills, 0, 1, ... * # of windmills, ... * len(valid_indxs)-1]
        
        
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        s_begin = self.valid_indxs[self.full_indx_row[index]]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len
        
        #map the index (range [0, # windmills * # of valid rows]) to station idx (range [0,63])
        station = self.full_indx_col[index]
        
        seq_x = self.data_x[s_begin:s_end, :, station] #shape: (look-back window, # of features)
        seq_y = self.data_x[r_begin:r_end, :, station] #shape: (pred_len + label_len, # of features)
        
        seq_x_mark = self.data_stamp[s_begin:s_end] #shape: (look-back window, # of time features)
        seq_y_mark = self.data_stamp[r_begin:r_end] #shape: (pred_len + label_len, # time of features)
        #Note: the pred_len + label_len dim comes from the fact that:
        #   (seq_len + pred_len) - (seq_len - label_len) #from the initial defn. of s- and r- begin end
        #   simplifies to pred_len + label_len
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    
    def __len__(self):
        if not self.all_stations:
            return len(self.valid_indxs)
        else:
            return self.data_indxs[self.valid_indxs, :].sum()
    
    

    ####NOTE: need to ensure this works as intended
    #Assumes inputs of shape [nodes, seq, feats] or [seq, feats]
    def inverse_transform(self, data):
        num_input_feats = data.shape[-1]
        if num_input_feats != len(self.scaler.scale_):
            data = np.concatenate([np.zeros([*data.shape[:-1], len(self.scaler.scale_) - data.shape[-1]]), data], -1)
        data = self.scaler.inverse_transform(data)

        if num_input_feats != len(self.scaler.scale_):
            data = data[..., -num_input_feats:]

        return data
    
    
    

class Dataset_wind_data_graph(Dataset):
    def __init__(self, root_path: str,              #path to root dir of clean data
                 dataset_size: str,
                 dataset_features: str,             #["one_feature", "subset_features", "all_features"]
                 flag: str = "train",               #["train", "val", "test"] - I think?
                 size : list[int, int, int] = None, #[seq_len, label_len, pred_len]
                 features: str = "S",               #S: univariate, M: multivariate
                 file_name: str = "wind_data.csv",  #name of file to read
                 target: str = None,                #name of station/windmill to target
                 scale: bool = True,                #
                 timeenc: int = 0,                  #
                 freq: str = "1h",                  #
                 subset: bool = False,              #
                 n_closest: int | None = None,      #
                 data_step: int = 5,
                 min_num_nodes: int = 2,
                 **_                                #allow for the function to take additional arguments - won't be used
                 ) -> None:
        
               
        assert flag in ["train", "test", "val"]
        flag_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = flag_map[flag]
        self.flag = flag
       
        #(notation used in paper)
        self.seq_len = size[0]      # S (the look-back window)
        self.label_len = size[1]    # L (the # of elements from the encoder inputs to also use in decoder input)
        self.pred_len = size[2]     # P (prediction horizon)
        
        self.total_seq_len = self.seq_len + self.pred_len
        assert self.label_len <= self.seq_len      
        
        
        self.data_step = data_step
        self.features = features
        self.min_num_nodes = min_num_nodes  #minimum number of nodes in a graph
        self.target = target                #if there is a station we want to always have in data (i.e. a target station)
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path
        self.dataset_size = dataset_size
        self.dataset_features = dataset_features
        self.file_name = file_name
        # Set to None if we want to have fully connected graphs. If n_closest = 3, then we only allow the three
        # closest nodes to send to a particular node.
        self.n_closest = n_closest
        
        #NOTE: should re-implement this 
        # If we want to only consider a subset of the stations. Just change the names to change the stations we want
        # to consider or None if we want to predict for all stations.
        if subset:
            #self.subset = [
            #    "570715000000030505",
            #    "",
            #    "",
            #    "",
            #]
            #self.subset = [
            #    'SNORREA',
            #    'SNORREB',
            #    'VISUNDFELTET',
            #    'KVITEBJØRNFELTET',
            #    'HULDRAFELTET',
            #    'VESLEFRIKKA',
            #    'OSEBERGC',
            #    'BRAGE',
            #    'OSEBERGSØR',
            #    'TROLLB',
            #    'GJØAFELTET'
            #]
            self.subset = None
        else:
            self.subset = None
        
        
        self.__read_data__()
        
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        _path = os.path.join(self.root_path, self.dataset_size, self.dataset_features, self.flag, self.file_name)        
        df_raw = pd.read_csv(_path, header=[0,1])
        
        
        # dict with indices of windmills - {GSRN: idx}
        self._stations = {s: i for i, s in enumerate(df_raw.columns.get_level_values(1).unique())}
        self._stations_inv = {v: k for k, v in self._stations.items()}
    
        #load node and edge data
        edge_feats_path = os.path.join(self.root_path, self.dataset_size, "edge_feats.csv")
        edge_feats = pd.read_csv(edge_feats_path, header=[0,1], index_col=0)
        node_path = os.path.join(self.root_path, self.dataset_size, "node_info.csv")
        node_info = pd.read_csv(node_path, dtype={"GSRN": str, "lat": float, "lon": float})
        self.node_info = node_info
        #only keep edge features of target node
        self.edge_feats = edge_feats[edge_feats.columns[edge_feats.columns.get_level_values(0) == self.target]]
        self.edge_feats.columns = self.edge_feats.columns.get_level_values(1)
    
        #uni- vs multi-variate (NOTE: SAME AS REGULAR WIND DATA)
        if self.features == "M" or self.features == "MS":
            #Mask out time (keep all other features)
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) != "TIME_UTC"]
            df_data = df_raw[cols_data]
            #Ensure that VAERDI (production) is the last column
            assert (df_raw.columns.get_level_values(0).unique()[-1:] == ["VAERDI"]).all()
        elif self.features == "S":
            #Only keep production data
            cols_data = df_raw.columns[df_raw.columns.get_level_values(0) == "VAERDI"]
            df_data = df_raw[cols_data]
    
        #Scaling (NOTE: SAME AS REGULAR WIND DATA)
        if self.scale:
                   
            #fit scaler based only on training data + only keep relevant cols (as per above if-else-statement)
            if self.flag != "train":
                train_data = pd.read_csv(_path.replace(self.flag, "train"), header=[0,1])
                train_data = train_data[cols_data]
                self.scaler.fit(train_data.stack(future_stack=True).values)
                del train_data #free up memory -> train_data is no longer used
            else:
                self.scaler.fit(df_data.stack(future_stack=True).values)
            
            #from (2621, 1472) --> (2621, 23, 64), that is (#rows, #features, #GSRN). note that 23*64 = 1472
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
            data = np.stack([self.scaler.transform(data[..., i]) for i in range(data.shape[-1])], -1) #apply the scaler for each individual turbine
        else:
            data = df_data.values.reshape(df_data.shape[0], df_data.columns.get_level_values(0).nunique(), -1)
    
        #select data only from the specified subset of nodes
        if self.subset is not None:
            subset_indxs = [self._stations[s] for s in self.subset]
            data = data[..., subset_indxs]
        
        
    
        #Find missing entries (NOTE: SAME AS REGULAR WIND DATA)
        nan_indxs = [np.where(np.isnan(data[..., i]).any(axis=1))[0] for i in range(data.shape[-1])] #list of arrays (one array for each windmill) --> each array represents the row idx with ANY missing value (across all feature cols)
        nan_indxs = [np.unique(np.concatenate([np.array([0]), nan_indxs[i], np.array([data[..., 0].shape[0] - 1])])) #create new array: [0, <..the idx of all rows containing any missing value (from above)..>, # of rows -1] --> basically just adds first and last value to NaN list
                     for i in range(len(nan_indxs))]  
        #nan_indxs is a list of arrays with # of arrays = # of windmills - each array contains the idx's of rows with at least a single NaN value      
        #eg ([0, 2620], ... * # of windmills (64))

        # Find the slices which result in valid sequences without NaNs (NOTE: SAME AS REGULAR WIND DATA)
        valid_slices = [np.where((nan_indxs[i][1:] - nan_indxs[i][:-1] - 1) >= self.total_seq_len)[0]
                        for i in range(len(nan_indxs))]
        valid_slices = [np.vstack([nan_indxs[i][valid_slices[i]] + 1, nan_indxs[i][valid_slices[i] + 1] - 1]).T
                        for i in range(len(nan_indxs))]
        #valid_slices is once again a list of arrays with # of arrays = # of windmills - now each array contains start and end idx's of rows with NO NaN values
        #eg ([1, 2619], ... * # of windmills (64))

        # Now, construct an array which contains the valid start indices for the different sequences
        #(NOTE: SAME AS REGULAR WIND DATA)
        data_indxs = []
        #for each windmill
        for i in range(len(valid_slices)):
            #create array of len: (# of rows - total_seq_len + 1) filled with False: ([False, ... * (# of rows - total_seq_len + 1)])
            # subtract total_seq_len as this gives last possible start index of the sliding window
            start_indxs = np.zeros(data.shape[0] - self.total_seq_len + 1, dtype='bool')
            
            #get start and end idx of valid slices
            for s, e in valid_slices[i]:
                #create a range from (start idx) to (end idx - total_seq_len)
                indxs_i = np.arange(s, e - self.total_seq_len + 2, 1)
                #turn False value of all valid rows into True
                start_indxs[indxs_i] = True
            data_indxs.append(start_indxs)
        #turn list of arrays to 2d array - [array(False, True, ... True, False), ... * # of windmills] --> (2552, # of windmills)
        self.data_indxs = np.stack(data_indxs, -1)
        #eg [[False False False ... False False False]
        #   [ True  True  True ...  True  True  True]
        #   ...
        #   [ True  True  True ...  True  True  True]
        #   [ True  True  True ...  True  True  True]
        #   [False False False ... False False False]]
        #   shape: (2552, 64), where 2552 = 2621 - 70(total_seq_len) + 1
        ######################################
        
                
        #construct time array 
        assert df_raw[["TIME_UTC"]].all(0).all()
        df_stamp = df_raw[["TIME_UTC"]].iloc[:, :1]
        df_stamp.columns = ["time"]
        df_stamp['time'] = pd.to_datetime(df_stamp.time)

        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.time.apply(lambda row: row.month)
            df_stamp["day"] = df_stamp.time.apply(lambda row: row.day)
            df_stamp["weekday"] = df_stamp.time.apply(lambda row: row.weekday())
            df_stamp["hour"] = df_stamp.time.apply(lambda row: row.hour)
            df_stamp["minute"] = df_stamp.time.apply(lambda row: row.minute)    #NOTE: redundant as max 1h resolution.
            data_stamp = df_stamp.drop(["time"], axis=1).values
        elif self.timeenc == 1:
            raise NotImplementedError("Time encoding has yet to be implemented")
        else:
            raise ValueError("Pass timeenc as either 0 or 1")

        
        self.data_x = data              #the full dataset (except time)
        self.data_stamp = data_stamp    #the time dataset
        
        #define all valid starting points of the sliding window
        self.valid_indxs = np.where(self.data_indxs.sum(1) >= self.min_num_nodes)[0] #index slices for the data
        self.valid_indxs = self.valid_indxs[::self.data_step] #apply the data_step size -> only look at every data_step'th datapoint
        
        # Find the n_closest number of nodes for every node. Use euclidean distance from scaled distances.
        if self.n_closest is not None:
            #if subset is specified, keep only the node features for those nodes, otherwise keep features for all nodes
            sub_node_info = self.node_info[self.node_info['GSRN'].isin(self.subset)] if self.subset is not None else self.node_info
            
            #using fit & transform of the MinMaxScaler to scale the lat/lon features into the features slat/slon
            latlon_scaler = MinMaxScaler()
            latlon_scaler.fit(sub_node_info[['lat', 'lon']].values)
            sub_node_info[['slat', 'slon']] = latlon_scaler.transform(sub_node_info[['lat', 'lon']].values)

            connectivity = {}
            #for each row (aka. node)
            for i, row_i in sub_node_info.iterrows():
                
                #compute euclidean distance to all nodes (inclduing to itself)
                dists = np.array(sub_node_info.apply(
                    lambda row: np.sqrt((row['slat'] - row_i.slat) ** 2 + (row['slon'] - row_i.slon) ** 2),
                    axis=1).to_list())
                
                #Add the GSRN of the current node to the connectivity dict as a key and a
                # list of all other nodes, sorted by the shortest dist to the current node,
                # as values
                connectivity[row_i['GSRN']] = sub_node_info.GSRN.iloc[np.argsort(dists)].values
            
            #Turn connectivity dict into df with node idx (0, ..., # nodes) as columns
            # with the rows within each column sorted by the distance to the corresponding columns
            # closest node is first row, etc. 
            connectivity = pd.DataFrame(connectivity)
            self.connectivity = connectivity.apply(lambda col: col.map(self._stations), axis=0)
            self.connectivity.columns = [self._stations[st] for st in self.connectivity.columns]

            self.connectivity = [
                self.connectivity.columns.values,   #array of nodes (0,..., # of nodes)
                self.connectivity.values            #array with shape (64,64) -> col = node id, row sorted by dist to node id (closest first)
            ]
        
        
        #create graph structure
        edge_feats = []
        senders = []    
        receivers = []
        #for each turbine (GSRN), node i
        for rec_i, stat_i in enumerate(self._stations.keys()):
            
            #extract node feature for node i
            info_i = self.node_info[self.node_info['GSRN'] == stat_i]
            
            #for each turbine (GSRN), node j
            for send_i, stat_j in enumerate(self._stations.keys()):
                receivers.append(rec_i)
                senders.append(send_i)
                
                #Create edge between node i and node j (will also create self-loop)
                info_j = self.node_info[self.node_info['GSRN'] == stat_j]
                
                #compute difference in lat/lon for node i and node j and add as edge_feature
                dlat = info_i.lat.iloc[0] - info_j.lat.iloc[0]
                dlon = info_i.lon.iloc[0] - info_j.lon.iloc[0]
                edge_feats.append([dlat, dlon])

        
        self.graph_struct = {
            'nodes': None,                          # [NxSxD]
            'edges': np.array(edge_feats),          # [N^2, 2]
            'senders': np.array(senders),           # [N^2,]
            'receivers': np.array(receivers),       # [N^2,]
            'station_names': self._stations.keys(),
        }
        
        
    
    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
        
        s_begin = self.valid_indxs[index]   #start of input sequence
        s_end = s_begin + self.seq_len      #end of input sequence
        r_begin = s_end - self.label_len    #start of output sequence - overlap of size L (label_len)
        r_end = s_end + self.pred_len       #end of output sequence
        
        #get idx of all stations active a this timestep
        stations = np.where(self.data_indxs[s_begin, :])[0] 
        
        #get input and output sequences for the selected stations
        seq_x = self.data_x[s_begin:s_end, :, stations] #(input time-range, # features, # turbines)
        seq_y = self.data_x[r_begin:r_end, :, stations] #(output time-range, # features, # turbines)
        
        
        #Get GSRN of the stations with available data
        if self.subset is not None:
            station_names = np.array(self.subset)[stations]
            stations = np.array([self._stations[s] for s in station_names])
        else:
            station_names = [self._stations_inv[i] for i in stations]
        
        
        #################
        ##NOTE: need to check this again...
        #################
        if self.n_closest is not None:
            def select_relevant_neighbors(col):
                """for each column/node with available data (<stations>), select the 1+n_closest nodes 
                (1+ since there are self-loops) 
                """
                return col[np.isin(col, stations)][:min(1 + self.n_closest, len(stations))]

            def get_edges_to_receiver_from_senders(col, col_name):
                """find edge indices where receiver is col_name and sender is in col"""
                col = np.where(
                    np.stack(
                        [(self.graph_struct['receivers'] == col_name),  #bool mask -> is receiver current node (col_name)
                         np.isin(self.graph_struct['senders'], col)]    #bool mask -> is sender in n_closest (col)
                    ).all(0)    #logical AND across both previous bool masks
                )[0]
                return col

            connect = self.connectivity[1][:, np.where(np.isin(self.connectivity[0], stations))[0]] #ensure connectivity is only kept among nodes with data avialable at given timestep
            connect = np.apply_along_axis(select_relevant_neighbors, axis=0, arr=connect)           #create array size (n_closest+1, stations)
            
            #Allow each receiver node to only have n_closest+1 incoming edges --> from the n_closest closest nodes + itself
            keep_edges = np.concatenate([get_edges_to_receiver_from_senders(connect[:, i], s) for i, s in
                                          enumerate(self.connectivity[0][np.isin(self.connectivity[0], stations)])])

        else:
            #include all edges between the nodes with available data
            keep_edges = np.where(np.stack([np.isin(self.graph_struct['senders'], stations),
                                            np.isin(self.graph_struct['receivers'], stations)]).all(0))[0]
        
        #remap global station/turbine indices to the local nodes indices
        graph_mapping = dict(zip(stations, np.arange(len(stations))))
        senders = np.vectorize(graph_mapping.get)(self.graph_struct['senders'][keep_edges])
        receivers = np.vectorize(graph_mapping.get)(self.graph_struct['receivers'][keep_edges])
        edge_feats = self.graph_struct['edges'][keep_edges]
        
        #graph in- and output for the GNNs
        graph_x = {
            'nodes': seq_x.transpose(2, 0, 1),       # [# nodes, seq_len, # features]
            'edges': edge_feats,                     # [(# nodes)^2, 2] | [(# nodes)*n_closest, 2] (if n_closest != None)
            'senders': senders,                      # [(# nodes)^2,] | [(# nodes)*n_closest,] (if n_closest != None)
            'receivers': receivers,                  # [(# nodes)^2,] | [(# nodes)*n_closest,] (if n_closest != None)
            'station_names': station_names,
        }
        graph_y = {
            'nodes': seq_y.transpose(2, 0, 1),      # [# nodes, pred_len + label_len, # features]
            'edges': np.array(edge_feats),          # same as graph_x
            'senders': np.array(senders),           # same as graph_x
            'receivers': np.array(receivers),       # same as graph_x
            'station_names': station_names,
        }

            
        seq_x_mark = self.data_stamp[s_begin:s_end] #shape: (look-back window, # of time features)
        seq_y_mark = self.data_stamp[r_begin:r_end] #shape: (pred_len + label_len, # time of features)
        #Note: the pred_len + label_len dim comes from the fact that:
        #   (seq_len + pred_len) - (seq_len - label_len) #from the initial defn. of s- and r- begin end
        #   simplifies to pred_len + label_len
        
        return graph_x, graph_y, seq_x_mark, seq_y_mark
    
    
    def __len__(self):
        return len(self.valid_indxs)
    
    

    # ####NOTE: need to ensure this works as intended
    # Assumes non-graph (i.e. mainly used for the outputs...)
    #  Inputs should be either of shape [nodes, seq_len, feats] or [seq_len, feats]
    def inverse_transform(self, data):
        num_input_feats = data.shape[-1]
        if num_input_feats != len(self.scaler.scale_):
            data = np.concatenate([np.zeros([*data.shape[:-1], len(self.scaler.scale_) - data.shape[-1]]), data], -1)
        data = self.scaler.inverse_transform(data)

        if num_input_feats != len(self.scaler.scale_):
            data = data[..., -num_input_feats:]

        return data



#NOTE: NEED TO CHECK AGAIN
# Custom collate function to graph samples into a batch
def collate_graph(batch):
    graph_x, graph_y, seq_x_mark, seq_y_mark = [[d[i] for d in batch] for i in range(len(batch[0]))]
    sizes_add = np.cumsum([0, *[g['nodes'].shape[0] for g in graph_x][:-1]])
    x = {
        'nodes': np.concatenate([g['nodes'] for g in graph_x], 0),
        'edges': np.concatenate([g['edges'] for g in graph_x], 0),
        'senders': np.concatenate([g['senders'] + start_i for g, start_i in zip(graph_x, sizes_add)]),
        'receivers': np.concatenate([g['receivers'] + start_i for g, start_i in zip(graph_x, sizes_add)]),
        'n_node': np.array([g['nodes'].shape[0] for g in graph_x]),
        'n_edge': np.array([g['edges'].shape[0] for g in graph_x]),
        'graph_mapping': np.stack([sizes_add, np.cumsum([g['nodes'].shape[0] for g in graph_x])], -1),
        'station_names': np.concatenate([g['station_names'] for g in graph_x]),
    }
    y = {
        'nodes': np.concatenate([g['nodes'] for g in graph_y], 0),
        'edges': np.concatenate([g['edges'] for g in graph_y], 0),
        'senders': np.concatenate([g['senders'] + start_i for g, start_i in zip(graph_y, sizes_add)]),
        'receivers': np.concatenate([g['receivers'] + start_i for g, start_i in zip(graph_y, sizes_add)]),
        'n_node': np.array([g['nodes'].shape[0] for g in graph_x]),
        'n_edge': np.array([g['edges'].shape[0] for g in graph_x]),
        'graph_mapping': np.stack([sizes_add, np.cumsum([g['nodes'].shape[0] for g in graph_x])], -1),
        'station_names': np.concatenate([g['station_names'] for g in graph_y]),
    }

    seq_x_mark = np.stack(seq_x_mark, 0)
    seq_y_mark = np.stack(seq_y_mark, 0)

    return x, y, torch.tensor(seq_x_mark), torch.tensor(seq_y_mark)