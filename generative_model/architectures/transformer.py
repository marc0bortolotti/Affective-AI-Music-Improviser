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
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layers for input and output sequences
        self.src_embedding = nn.Embedding(vocab_size[0], d_model)
        self.tgt_embedding = nn.Embedding(vocab_size[1], d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_length)
        
        # Transformer model (Encoder-Decoder)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        
        # Linear layer to map the output to vocabulary size for predictions
        self.fc_out = nn.Linear(d_model, vocab_size[1])

        self.max_seq_length = max_seq_length
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed the source and target sequences
        src = self.positional_encoding(self.src_embedding(src))

        if tgt is None:
            # Generate a tensor of zeros or a fixed token shape.
            tgt = torch.zeros(src.size(0), self.max_seq_length, dtype=torch.long).to(src.device) 
        
        tgt = self.positional_encoding(self.tgt_embedding(tgt))
        
        # Pass through the Transformer
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                  src_key_padding_mask=src_padding_mask,
                                  tgt_key_padding_mask=tgt_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        
        # Final linear layer to project to the vocab size
        return self.fc_out(output)

# Helper function to generate a square subsequent mask for target sequence (used in decoding to avoid peeking at future tokens)
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == '__main__':

    import time

    # Example usage
    vocab_size = [877, 16447]        # Vocabulary size for input/output sequences
    d_model = 512             # Embedding dimension and model size
    nhead = 8                 # Number of attention heads
    num_encoder_layers = 6    # Number of encoder layers
    num_decoder_layers = 6    # Number of decoder layers
    dim_feedforward = 2048    # Feedforward network dimension
    max_seq_length = 64      # Max sequence length
    dropout = 0.1             # Dropout rate

    # Create the model
    model = TransformerModel(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_seq_length, dropout)

    # Example source and target sequences (batch size = 32, sequence length = 100)
    src = torch.randint(0, vocab_size[0], (max_seq_length, 32))  # Source: (sequence length, batch size)
    tgt = torch.randint(0, vocab_size[1], (max_seq_length, 32))  # Target: (sequence length, batch size)

    # Generate target mask to prevent looking at future tokens
    tgt_mask = generate_square_subsequent_mask(max_seq_length)

    # Forward pass through the model
    start = time.time()
    output = model(src, tgt, tgt_mask=tgt_mask)
    print("Time taken for forward pass: ", time.time() - start)
    print(output.shape)  # Output: (sequence length, batch size, vocab_size)

    # print the total dimension of the model
    print(sum(p.numel() for p in model.parameters()))