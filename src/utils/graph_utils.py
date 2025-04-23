from src.layers.graphs import GraphsTuple

import torch

def data_dicts_to_graphs_tuple(graph_dicts: dict, device=None) -> GraphsTuple:
    for k, v in graph_dicts.items():
        if k in ['senders', 'receivers', 'n_node', 'n_edge', 'graph_mapping']:
            graph_dicts[k] = torch.tensor(v, dtype=torch.int64)
        elif k == 'station_names':
            continue
        else:
            graph_dicts[k] = torch.tensor(v, dtype=torch.float32)
        if device is not None:
            graph_dicts[k] = graph_dicts[k].to(device)
    return GraphsTuple(**graph_dicts)

