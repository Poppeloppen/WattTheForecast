import argparse
import torch
import torch.nn as nn
from typing import Any

class Model(nn.Module):
    """
    Persistence model
    """
    
    def __init__(self, configs: argparse.Namespace) -> None:
        super().__init__()
        self.label_len = configs.label_len  #L: start token length (default=48)
        self.pred_len = configs.pred_len    #P: # of steps to predict into the future (default=6)
        self.seq_len = configs.seq_len      #S: look-back window (default=64)
        self.output_size = configs.c_out    #... (default=1)
        
    def forward(self, x_enc: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        #x_enc: the dataset (excluding time info) - shape: (batch_size/32, seq_len/60, enc_in/23)
        #*_: unused *args 
        #**__: unused **kwargs
        
        #extract the most recent observation of the target variable (which is the last feature)
        # shape (batch_size/32, seq_len/60, enc_in/23) --> (batch_size/32, 1, c_out/1)
        outputs = x_enc[:, -1:, -self.output_size:] #Note the slicing (:) is needed to keep dimension

        #repeat the last observation pred_len times
        # shape (batch_size/32, 1, c_out/1) --> (batch_size/32, pred_len/6, c_out/1)
        outputs = outputs.repeat(1, self.pred_len, 1)

        return outputs