import argparse
from torch.utils.data import DataLoader

from src.data.data_loader import Dataset_wind_data, Dataset_wind_data_graph, collate_graph

data_dict = {
    "Wind" : Dataset_wind_data,
    "WindGraph" : Dataset_wind_data_graph
}


def data_provider(args: argparse.Namespace, flag: str) -> list[Dataset_wind_data, DataLoader]:
    Data = data_dict[args.data]
    
    # See the __read_data__() func from Dataset_wind_data in ./data_loader.py
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == "test":
        shuffle_flag = False            
        drop_last = True                
        batch_size = args.batch_size    
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        
    data_set = Data(
        root_path = args.root_path,
        dataset_size = args.dataset_size,
        dataset_features = args.dataset_features,
        file_name = args.file_name,
        flag = flag,
        size = [args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        timeenc = timeenc,
        freq = freq,
        n_closest = args.n_closest,         
        all_stations = args.all_stations,
        data_step = args.data_step,
        min_num_nodes = args.min_num_nodes
    )
    
    print(flag, len(data_set))
    
    if args.data == 'WindGraph':
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=collate_graph
        )
        
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,          
            shuffle=shuffle_flag,           #whether to have the data reshuffled at every epoch (default=False)    
            num_workers=args.num_workers,   #how many subprocesses to use for data loading. ``0`` means that the data will be loaded in the main process. (default=0)
            drop_last=drop_last             #whether to drop the last incomplete batch, if the dataset size is not divisible by the batch size. (default=False) 
        )
    
    
    return data_set, data_loader



