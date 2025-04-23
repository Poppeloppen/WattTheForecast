import torch 
import torch.nn as nn
import torch.nn.functional as F

class MLPLayer(nn.Module):
    def __init__(self, d_model: int,
                 d_ff: ...,
                 kernel_size: int = 1,
                 dropout: float = 0.,
                 activation: str = 'relu',
                 res_con: bool = True
                 ) -> None:
        super(MLPLayer, self).__init__()
        
        self.kernel_size = kernel_size
        if self.kernel_size != 1:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=(kernel_size-1))
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size, padding=(kernel_size-1))
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.res_con = res_con

    def forward(self, x: ...) -> ...:
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        if self.kernel_size != 1:
            y = y[..., 0:-(self.kernel_size - 1)]
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if self.kernel_size != 1:
            y = y[:, 0:-(self.kernel_size - 1), :]
        if self.res_con:
            return self.norm2(x + y)
        else:
            return y