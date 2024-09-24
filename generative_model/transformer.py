import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention, value)
        return output, attention

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.linear_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.linear_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.linear_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)
        
        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc_out(scaled_attention)
        return output, attention_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # Self-attention
        attn_output, _ = self.self_attn(src, src, src, mask)
        src = self.layernorm1(src + self.dropout(attn_output))
        # Feed-forward
        ff_output = self.feed_forward(src)
        src = self.layernorm2(src + self.dropout(ff_output))
        return src

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src = self.embedding(src) * math.sqrt(src.size(-1))
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention in decoder
        tgt_attn, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.layernorm1(tgt + self.dropout(tgt_attn))

        # Cross-attention with encoder output
        cross_attn, _ = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.layernorm2(tgt + self.dropout(cross_attn))

        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = self.layernorm3(tgt + self.dropout(ff_output))
        return tgt

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.embedding(tgt) * math.sqrt(tgt.size(-1))
        tgt = self.pos_encoder(tgt)
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        output = self.fc_out(tgt)
        return output

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, d_ff=2048, num_encoder_layers=6, num_decoder_layers=6, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_encoder_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_decoder_layers, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask)
        return output

# Example usage:
src_vocab_size = 10000  # Source vocabulary size
tgt_vocab_size = 10000  # Target vocabulary size
d_model = 512  # Embedding size and hidden dimension
num_heads = 8  # Number of attention heads
d_ff = 2048  # Feed-forward hidden dimension
num_layers = 6  # Number of encoder/decoder layers

model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, num_layers)

# Example source and target input (batch_size, sequence_length)
src = torch.randint(0, src_vocab_size, (32, 20))  # Random source input
tgt = torch.randint(0, tgt_vocab_size, (32, 20))  # Random target input

output = model(src, tgt)
print(output.shape)  # Output shape: (batch_size, sequence_length, vocab_size)
