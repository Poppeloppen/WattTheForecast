#NOTE: look into details

import argparse
import torch
import torch.nn as nn
from typing import Any
from src.layers.LSTM_EncDec import Encoder, Decoder
from src.layers.Embed import DataEmbedding
import random


class Model(nn.Module):
    """
    LSTM in Encoder-Decoder
    """
    def __init__(self, configs: argparse.Namespace) -> None:
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.enc_layers = configs.e_layers
        self.dec_layers = configs.d_layers

        self.train_strat_lstm = configs.train_strat_lstm

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_size = configs.c_out
        assert configs.label_len >= 1

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, kernel_size=configs.kernel_size, pos_embed=False)
        # self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout, kernel_size=configs.kernel_size, pos_embed=False)

        self.encoder = Encoder(d_model=self.d_model, num_layers=self.enc_layers, dropout=configs.dropout)
        self.decoder = Decoder(output_size=configs.c_out, d_model=self.d_model,
                               dropout=configs.dropout, num_layers=self.dec_layers)

    def forward(self, x_enc: torch.Tensor,
                x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor,
                x_mark_dec: torch.Tensor,
                teacher_forcing_ratio: float | None = None,
                batch_y: torch.Tensor = None,
                **_: Any) -> torch.Tensor:
        #x_enc: the dataset (excluding time info) - shape: (batch_size/32, seq_len/60, enc_in/23)
        #x_mark_enc: the time data - shape: (batch_size/32, seq_len/60, <# of time features>/5)
        #x_dec: 
        #x_mark_dec: 
        #teacher_forcing_ratio: float representing ....
        #batch_y:  
        #**_: unused **kwargs

        
        if self.train_strat_lstm == 'mixed_teacher_forcing' and self.training:
            assert teacher_forcing_ratio is not None
            target = batch_y[:, -self.pred_len:, -self.output_size:]

        # value + temporal embedding (positional embedding is not used), shape: (batch_size, seq_len, d_model)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        
        enc_out, enc_hid = self.encoder(enc_out)

        if self.enc_layers != self.dec_layers:
            assert self.dec_layers <= self.enc_layers
            enc_hid = [hid[-self.dec_layers:, ...] for hid in enc_hid]

        dec_inp = x_dec[:, -(self.pred_len + 1), -self.output_size:]
        dec_hid = enc_hid

        outputs = torch.zeros((x_enc.shape[0], self.pred_len, self.output_size)).to(enc_out.device)

        if not self.training or self.train_strat_lstm == 'recursive':
            for t in range(self.pred_len):
                dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                outputs[:, t, :] = dec_out
                dec_inp = dec_out
        else:
            if self.train_strat_lstm == 'mixed_teacher_forcing':
                for t in range(self.pred_len):
                    dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                    outputs[:, t, :] = dec_out
                    if random.random() < teacher_forcing_ratio:
                        dec_inp = target[:, t, :]
                    else:
                        dec_inp = dec_out

        return outputs  # [B, L, D]
