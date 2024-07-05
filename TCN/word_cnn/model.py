import numpy as np
import torch
from torch import nn
import sys
sys.path.append("../../")
from TCN.tcn import TemporalConvNet


class TCN(nn.Module):

    def __init__(self, input_size, embedding_size, output_size, num_channels, emphasize_eeg=False, 
                 kernel_size=2, dropout=0.3, emb_dropout=0.1, tied_weights=False):
        
        super(TCN, self).__init__()

        if emphasize_eeg:
            self.encoder = nn.Embedding(input_size, embedding_size, padding_idx=0) # padding_idx is the index of the token to ignore in weight updates
        else:
            self.encoder = nn.Embedding(input_size, embedding_size)

        self.tcn = TemporalConvNet(embedding_size, num_channels, kernel_size, dropout=dropout)

        # if emphasize_eeg:
        #     num_channels[-1] += 1
        #     self.weights_layer = nn.Linear(num_channels[-1], num_channels[-1])
        #     weights = np.identity(num_channels[-1])
        #     weights[-1] =  
        #     self.weights_layer.weight = torch.Tensor(np.identity(num_channels[-1]), requires_grad=False)
        #     self.weights_layer.bias = torch.Tensor(np.zeros(num_channels[-1]), requires_grad=False)

        self.decoder = nn.Linear(num_channels[-1], output_size)

        if tied_weights:
            if num_channels[-1] != embedding_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
            
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.emphasize_eeg = emphasize_eeg
        self.init_weights()

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

        if self.emphasize_eeg:
            with torch.no_grad():
                self.encoder.weight[0] = torch.ones(self.encoder.weight[0].shape)*0.1

    def forward(self, input, eeg_class=None):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""
        emb = self.drop(self.encoder(input))
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)
        y = self.decoder(y)
        return y.contiguous()

