import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, embedding_dim, nhead, num_layers, seq_length, dim_feedforward):
        super(MusicTransformer, self).__init__()
        self.in_embedding = nn.Embedding(in_vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(out_vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))
        self.transformer = nn.Transformer(d_model=embedding_dim, 
                                          nhead=nhead, 
                                          dim_feedforward=dim_feedforward,
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers,
                                          batch_first=True)
        self.fc_out = nn.Linear(embedding_dim, out_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Class variable to store predictions
        self.predictions = None

        self.PARAMS = { 'in_vocab_size' : in_vocab_size,
                        'out_vocab_size' : out_vocab_size,
                        'embedding_dim' : embedding_dim,
                        'nhead' : nhead,
                        'num_layers' : num_layers,
                        'seq_length' : seq_length,
                        'dim_feedforward' : dim_feedforward,
                        }

    def size(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and add positional encoding
        src = self.in_embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt = self.out_embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]

        transformer_output = self.transformer(  src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                                src_key_padding_mask=src_padding_mask,
                                                tgt_key_padding_mask=tgt_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask)
        
        y = self.fc_out(transformer_output)  # Map to vocab size
        
        return y
    
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

