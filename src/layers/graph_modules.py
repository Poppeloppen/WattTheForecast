import torch 
import torch.nn as nn
from typing import Callable, Any
from src.layers.graphs import GraphsTuple
import src.layers.graph_blocks as blocks
from src.layers.Functionality import MLPLayer
from src.layers.MLP_Layer import MLPLayer


class GraphLayer(nn.Module):
    def __init__(self, edge_update_fn,
                 node_update_fn,
                 d_model: int,
                 edges_agg: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = blocks.unsorted_mean_agg,
                 num_node_series: int = 1,
                 num_edge_series: int = 1,
                 n_closest: int | None = None
                 ) -> None:       
        super(GraphLayer, self).__init__()
        self._edge_block = blocks.EdgeBlock(edge_update_fn, d_model, num_node_series=num_node_series,
                                            num_edge_series=num_edge_series)
        self._node_block = blocks.NodeBlock(node_update_fn, d_model, edges_agg=edges_agg,
                                            num_node_series=num_node_series,
                                            num_edge_series=num_edge_series,
                                            use_received_edges=True)
        self.n_closest = n_closest

    def forward(self, graph: GraphsTuple, **kwargs: Any) -> tuple[GraphsTuple, list[list, list]] | tuple[GraphsTuple, list[None, list]]:
        graph, attn_edges = self._edge_block(graph, **kwargs)
        graph, attn_nodes = self._node_block(graph, **kwargs)

        if self.n_closest != 0:
            return graph, [attn_edges, attn_nodes]
        else:
            return graph, [None, attn_nodes]



class GraphNetwork(nn.Module):
    def __init__(self, layers: list[GraphLayer],
                 norm_layer = None,
                 projection: nn.Linear| None = None,
                 mlp_out = None,
                 norm_layer2 = None,
                 **_: Any
                 ) -> None:
        super(GraphNetwork, self).__init__()
        
        print("layers type", type(layers))
        print("layers len", len(layers))
        print("layers type", type(layers[0]))
        print("norm_layer type", type(norm_layer))
        print("projection type", type(projection))
        print("mlp_out type", type(mlp_out))
        print("norm_layer2 type", type(norm_layer2))
        
        
        self._layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.norm2 = norm_layer2
        self.mlp_out = mlp_out
        self.projection = projection

    def forward(self, x: GraphsTuple, trend=None, **kwargs: Any) -> tuple[GraphsTuple, list]:
        attn = []
        for layer in self._layers:
            x, a_sa = layer(x, **kwargs)

            # Autoformer stuff
            if isinstance(x, list):
                assert len(x) == 2
                trend = trend + x[-1]
                x = x[0]

            attn.append(a_sa)

        if self.norm2 is not None:
            x_freq, x_trend = torch.unbind(x.nodes, dim=-1)
            x_freq = self.norm(x_freq)
            x_trend = self.norm2(x_trend)
            x = x.replace(nodes=x_freq+x_trend)
        else:
            if self.norm is not None:
                x = x.replace(nodes=self.norm(x.nodes))
        if len(x.nodes.shape) == 4:
            x_freq, x_trend = torch.unbind(x.nodes, dim=-1)
            x = x.replace(nodes=x_freq+x_trend)

        if self.mlp_out is not None:
            x = x.replace(nodes=self.mlp_out(x.nodes))

        if self.projection is not None:
            x = x.replace(nodes=self.projection(x.nodes))

        # Autoformer stuff
        if trend is not None:
            x = x.replace(nodes=x.nodes + trend)

        return x, attn
