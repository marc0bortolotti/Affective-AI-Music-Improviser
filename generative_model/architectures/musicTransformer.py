import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Define the Transformer-based model
class MusicTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, max_length):
        super(MusicTransformer, self).__init__()

        self.encoder_embedding = nn.Embedding(input_vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_length, embed_size))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=forward_expansion * embed_size)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_size, nhead=num_heads, dim_feedforward=forward_expansion * embed_size)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_size, output_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        # src: (batch_size, src_seq_len)
        # tgt: (batch_size, tgt_seq_len)
        
        # Add embeddings and positional encodings to source and target sequences
        src_emb = self.dropout(self.encoder_embedding(src) + self.positional_encoding[:, :src.size(1), :])
        tgt_emb = self.dropout(self.decoder_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :])

        # Encoder
        memory = self.encoder(src_emb)

        # Decoder
        out = self.decoder(tgt_emb, memory)

        # Final output layer
        output = self.fc_out(out)

        return output

# Example Dataset for Music Tokens
class MusicDataset(Dataset):
    def __init__(self, src_sequences, tgt_sequences):
        self.src_sequences = src_sequences
        self.tgt_sequences = tgt_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx], dtype=torch.long), torch.tensor(self.tgt_sequences[idx], dtype=torch.long)

# Hyperparameters
INPUT_VOCAB_SIZE = 877  # Source instrument vocabulary size
OUTPUT_VOCAB_SIZE = 16000  # Target instrument vocabulary size
EMBED_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 512  # Maximum sequence length
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# Generate synthetic dataset (for illustration purposes, in practice you'd use real music data)
def generate_synthetic_music_data(num_samples, seq_length, vocab_size):
    return [[torch.randint(0, vocab_size, (seq_length,)).tolist() for _ in range(num_samples)]]

# For demonstration, we are creating 12000 sequences of source and target tokens
NUM_SEQUENCES = 12000
SEQUENCE_LENGTH = 100  # Assuming sequences are of length 100

src_sequences = generate_synthetic_music_data(NUM_SEQUENCES, SEQUENCE_LENGTH, INPUT_VOCAB_SIZE)[0]
tgt_sequences = generate_synthetic_music_data(NUM_SEQUENCES, SEQUENCE_LENGTH, OUTPUT_VOCAB_SIZE)[0]

# Create Dataset and DataLoader
dataset = MusicDataset(src_sequences, tgt_sequences)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize the model
model = MusicTransformer(
    input_vocab_size=INPUT_VOCAB_SIZE,
    output_vocab_size=OUTPUT_VOCAB_SIZE,
    embed_size=EMBED_SIZE,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    forward_expansion=FORWARD_EXPANSION,
    dropout=DROPOUT,
    max_length=MAX_LENGTH
)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for batch_idx, (src, tgt) in enumerate(dataloader):
        # Shift target to the right for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Forward pass
        output = model(src, tgt_input)

        # Reshape output and target to fit into the loss function
        output = output.reshape(-1, OUTPUT_VOCAB_SIZE)
        tgt_output = tgt_output.reshape(-1)

        loss = criterion(output, tgt_output)
        epoch_loss += loss.item()

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss / len(dataloader):.4f}")

# Inference: Generating a new sequence for a target instrument
def generate_music(model, src_sequence, max_length, start_token):
    model.eval()
    src = torch.tensor(src_sequence).unsqueeze(0)  # Add batch dimension
    tgt = torch.tensor([start_token]).unsqueeze(0)  # Start with a start token
    
    for _ in range(max_length):
        # Forward pass
        with torch.no_grad():
            output = model(src, tgt)

        # Get the predicted next token (the one with the highest probability)
        next_token = output.argmax(dim=-1)[:, -1]

        # Append the predicted token to the target sequence
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)

        # Stop if the model predicts an end token (for simplicity, let's assume token 0 is <END>)
        if next_token.item() == 0:
            break

    return tgt.squeeze(0).tolist()

# Example of generating music
src_sequence = torch.randint(0, INPUT_VOCAB_SIZE, (SEQUENCE_LENGTH,)).tolist()  # Random input sequence for source instrument
start_token = 1  # Define start token for the target instrument (usually <START>)
generated_sequence = generate_music(model, src_sequence, max_length=100, start_token=start_token)

print("Generated sequence:", generated_sequence)
