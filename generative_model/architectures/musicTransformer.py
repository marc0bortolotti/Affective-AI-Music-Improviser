import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Hyperparameters
vocab_size = 128  # Assuming 128 possible tokens (e.g., MIDI notes)
embedding_dim = 256
nhead = 8  # Number of heads in the multi-head attention
num_layers = 6  # Number of Transformer layers
seq_length = 48  # 3 bars * 16 tokens/bar = 48 input tokens
output_length = 16  # Predict the next 16 tokens (next bar)
batch_size = 32
epochs = 100
learning_rate = 0.001

# Sample data generator (for demonstration)
def generate_dummy_data(num_sequences=10000, seq_length=48, output_length=16):
    data = np.random.randint(0, vocab_size, (num_sequences, seq_length + output_length))
    inputs = data[:, :seq_length]
    targets = data[:, seq_length:]
    return torch.LongTensor(inputs), torch.LongTensor(targets)

# Transformer Model for Music Generation
class MusicTransformer(nn.Module):
    def __init__(self, in_vocab_size, out_vocab_size, embedding_dim, nhead, num_layers, seq_length):
        super(MusicTransformer, self).__init__()
        self.in_embedding = nn.Embedding(in_vocab_size, embedding_dim)
        self.out_embedding = nn.Embedding(out_vocab_size, embedding_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_length, embedding_dim))
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, out_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        
        # Class variable to store predictions
        self.predictions = None

        self.PARAMS = { 'in_vocab_size' : in_vocab_size,
                          'out_vocab_size' : out_vocab_size,
                          'embedding_dim' : embedding_dim,
                          'nhead' : nhead,
                          'num_layers' : num_layers,
                          'seq_length' : seq_length
                        }

    def size(self): 
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        # Embed and add positional encoding
        src = self.in_embedding(src) + self.pos_encoder[:, :src.size(1), :]
        tgt = self.out_embedding(tgt) + self.pos_encoder[:, :tgt.size(1), :]
        
        # Transformer expects input in (sequence_length, batch_size, embed_dim) format
        src = src.permute(1, 0, 2)  # (seq_length, batch_size, embedding_dim)
        tgt = tgt.permute(1, 0, 2)  # (output_length, batch_size, embedding_dim)

        transformer_output = self.transformer(  src, tgt, src_mask=src_mask, tgt_mask=tgt_mask,
                                                src_key_padding_mask=src_padding_mask,
                                                tgt_key_padding_mask=tgt_padding_mask,
                                                memory_key_padding_mask=memory_key_padding_mask)
        
        output = self.fc_out(transformer_output)  # Map to vocab size

        # Save the output predictions in the class variable
        self.predictions = output.permute(1, 0, 2)  # Save in (batch_size, seq_length, vocab_size) format
        
        return self.predictions
    
def generate_square_subsequent_mask(sz):
    mask = torch.triu(torch.ones(sz, sz)) == 1
    mask = mask.transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Training and Generation Functions
def train_model(model, data_loader, criterion, optimizer, epochs=100):
    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in data_loader:
            inputs, targets = batch
            optimizer.zero_grad()

            # Split targets to get the correct input format for the decoder
            tgt_input = targets[:, :-1]  # Target input is all but last token
            tgt_output = targets[:, 1:]  # Target output is all but first token

            # Forward pass
            output = model(inputs, tgt_input)

            # Reshape to match the dimensions expected by CrossEntropyLoss
            output = output.view(-1, vocab_size)
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(data_loader)}')

def generate_music(model, input_sequence, max_length=16):
    model.eval()
    generated_sequence = input_sequence

    for _ in range(max_length):
        input_seq = torch.LongTensor(generated_sequence).unsqueeze(0)  # Add batch dimension
        output_seq = torch.LongTensor(generated_sequence[-16:]).unsqueeze(0)  # Decoder takes the last generated sequence

        # Get the model's prediction
        output = model(input_seq, output_seq)
        next_token = output.argmax(dim=-1)[:, -1].item()  # Get the predicted next token
        
        # Add predicted token to generated sequence
        generated_sequence.append(next_token)

    # Store the generated sequence in the model's class variable
    model.predictions = generated_sequence

    return generated_sequence

# Main Execution
if __name__ == "__main__":
    # Create dataset
    inputs, targets = generate_dummy_data()

    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss, and optimizer
    model = MusicTransformer(vocab_size, embedding_dim, nhead, num_layers, seq_length, output_length)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(model, data_loader, criterion, optimizer, epochs=epochs)

    # Generate music
    input_sequence = inputs[0].tolist()[:seq_length]  # Take first 48 tokens
    generated_sequence = generate_music(model, input_sequence)

    print("Generated Sequence: ", generated_sequence)

    # Access the saved predictions directly from the model class
    print("Predictions stored in model: ", model.predictions)
