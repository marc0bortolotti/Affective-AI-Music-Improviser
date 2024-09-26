import torch
import torch.nn as nn
import torch.optim as optim

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.PARAMS = { 'input_vocab_size' : input_vocab_size,
                        'output_vocab_size' : output_vocab_size,
                        'd_model' : d_model, # Embedding dimension and model size
                        'nhead' : nhead, # Number of attention heads
                        'num_encoder_layers' : num_encoder_layers,
                        'num_decoder_layers' : num_decoder_layers,
                        'dim_feedforward' : dim_feedforward,
                        'max_seq_length' : max_seq_length,
                        'dropout' : dropout,
                    }
        
        # Embedding layers for input and output sequences
        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(output_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        
        # Transformer model (Encoder-Decoder)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        
        # Linear layer to map the output to vocabulary size for predictions
        self.fc_out = nn.Linear(d_model, output_vocab_size)

        self.max_seq_length = max_seq_length

    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed the source and target sequences
        src = self.positional_encoding(self.src_embedding(src))

        if tgt is None:
            # Generate a tensor of zeros or a fixed token shape.
            tgt = torch.zeros(src.size(0), self.max_seq_length, dtype=torch.long).to(src.device) 
        
        tgt = self.positional_encoding(self.tgt_embedding(tgt))

        # Transformer expects input in (sequence_length, batch_size, embed_dim) format, we got (batch_size, sequence_length, embed_dim)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)
        
        # Pass through the Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Final linear layer to project to the vocab size
        y = self.fc_out(output)

        return y