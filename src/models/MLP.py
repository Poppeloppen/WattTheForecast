import argparse
import torch
import torch.nn as nn
from typing import Any

from src.layers.MLP_Layer import MLPLayer

class Model(nn.Module):
    def __init__(self, configs: argparse.Namespace) -> None:
        super(Model, self).__init__()
        self.seq_len = configs.seq_len                      #S: look-back window (default=60)
        self.pred_len = configs.pred_len                    #P: # of steps to predict into the future (default=6)
        self.input_features = configs.enc_in                # # of encoder input features (default=23)
        self.output_features = configs.c_out                #output size (default=1)
        self.d_model = configs.d_model                      #dimension of model (default=512)
        self.layers = configs.e_layers                      # # of encoder layers (default=2)
        self.output_attention = configs.output_attention    # whether to output attention in encoder (default=False)

        # Encoder
        self.mlp = nn.ModuleList(
            [
                MLPLayer(
                    input_size=self.d_model if i != 0 else (self.input_features * self.seq_len),
                    output_size=self.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    norm_layer='layer'
                )
                for i in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model, (self.output_features * self.pred_len), bias=True)


    def forward(self, x: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor | tuple[torch.Tensor, None]:
        #x: the dataset (excluding time info) - shape: (batch_size/32, seq_len/60, enc_in/23)
        #*_: unused *args 
        #**__: unused **kwargs
        
        # Reshape input 3d->2d ((batch_size/32, seq_len/60, enc_in/23) --> (batch_size/32, seq_len*enc_in/60*23=1380))
        outputs = x.reshape(x.shape[0], -1)

        # Pass through MLP (hidden layers) - result in the shape: (batch_size/32, d_model/512)
        for layer in self.mlp:
            outputs = layer(outputs)

        # Project from hidden layer to output ((batch_size/32, d_model/512) --> (batch_size/32, c_out*pred_len/1*6=6))
        outputs = self.projection(outputs)
        
        # Reshape to correct output ((batch_size/32, c_out*pred_len/1*6=6) --> (batch_size/32, pred_len/6, c_out/1))
        outputs = outputs.view(outputs.shape[0], self.pred_len, self.output_features)
        
        if self.output_attention:
            return outputs, None
        else:
            return outputs