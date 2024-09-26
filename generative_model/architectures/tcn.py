import torch
from torch import nn
from torch.nn.utils import weight_norm
from math import log2

'''
    IMPORTANT:
    to cover all the sequence of tokens k * d must be >= hidden units (see the paper)
    k = kernel_size
    d = dilation = 2 ^ (n_levels - 1)

    so levels = log2(hidden_units / kernel_size + 1) + 1
    '''

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):

    def __init__(self, input_vocab_size, embedding_size, output_vocab_size, hidden_units, seq_length, emphasize_eeg=False, feedback=False,
                 levels = None, kernel_size=3, dropout=0.45, emb_dropout=0.25, tied_weights=False):
        
        super(TCN, self).__init__()

        self.feedback = feedback
        self.prev_output = None
        self.seq_length = seq_length
        self.bar_length = int(self.seq_length / 4)

        if levels is None:
            levels = int(log2(hidden_units / kernel_size + 1) + 1)

        if feedback:
            input_vocab_size += output_vocab_size
            levels += 1
            hidden_units *= 2 
        
        num_channels = [hidden_units] * (levels - 1) + [embedding_size] # [192, 192, 192, 192, 192, 192, 20]

        self.PARAMS = { 'input_vocab_size' : input_vocab_size,
                        'output_vocab_size' : output_vocab_size,
                        'embedding_size' : embedding_size,
                        'levels' : levels,
                        'emphasize_eeg' : emphasize_eeg,
                        'hidden_units' : hidden_units,
                        'feedback' : feedback,
                        'dropout' : dropout,
                        'emb_dropout' : emb_dropout,
                        'kernel_size' : kernel_size,
                        'tied_weights' : tied_weights,
                        'seq_length' : seq_length
                    }

        if emphasize_eeg:
            self.encoder = nn.Embedding(input_vocab_size, embedding_size, padding_idx=0) # padding_idx is the index of the token to ignore in weight updates
        else:
            self.encoder = nn.Embedding(input_vocab_size, embedding_size)

        self.tcn = TemporalConvNet(embedding_size, num_channels, kernel_size, dropout=dropout)

        self.decoder = nn.Linear(num_channels[-1], output_vocab_size)

        if tied_weights:
            if num_channels[-1] != embedding_size:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
            
        self.drop = nn.Dropout(emb_dropout)
        self.emb_dropout = emb_dropout
        self.emphasize_eeg = emphasize_eeg
        self.init_weights()

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def init_weights(self):
        self.encoder.weight.data.normal_(0, 0.01)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.normal_(0, 0.01)

        if self.emphasize_eeg:
            with torch.no_grad():
                self.encoder.weight[0] = torch.ones(self.encoder.weight[0].shape)*0.1

    def forward(self, input):
        """Input ought to have dimension (N, C_in, L_in), where L_in is the seq_len; here the input is (N, L, C)"""

        # add mask to the last bar
        input_masked = torch.cat((input[:, :self.bar_length*3], torch.zeros([input.size(0), self.bar_length], dtype=torch.long, device = input.device)), dim = 1)

        # if feedback is enabled, concatenate the previous output to the input
        if self.feedback:

            # initialize the previous output if it is the first iteration
            if self.prev_output is None:
                self.prev_output = torch.zeros([input.size(0), input.size(1)], dtype=torch.long, device = input.device)
            
            # get the minimum batch size between the input and the previous output
            min_batch = min(input.size(0), self.prev_output.size(0))

            # concatenate the previous output to the input
            input_masked = torch.cat((input_masked[:min_batch, :], self.prev_output[:min_batch, :]), dim = 1)

        # add embeddings and apply dropout
        emb = self.drop(self.encoder(input))

        # apply TCN
        y = self.tcn(emb.transpose(1, 2)).transpose(1, 2)

        # apply linear layer
        y = self.decoder(y)

        # save the output for feedback
        if self.feedback:
            self.prev_output = torch.argmax(y, dim = 2)
            y = y[:, :self.seq_length]

        return y

    
