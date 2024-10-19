import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder_lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        self.PARAMS = { 
                        'hidden_dim': hidden_dim,
                        'latent_dim': latent_dim,
                        'vocab_size': vocab_size,
                        'embed_dim': embed_dim,
                    }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.embedding(x)  # Convert token indices to embeddings
        _, (h_n, _) = self.encoder_lstm(x)
        h_n = h_n[-1]  # Get the last hidden state
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def decode(self, z, seq_length):
        z = z.unsqueeze(1).repeat(1, seq_length, 1)  # Repeat z for the length of the sequence
        out, _ = self.decoder_lstm(z)
        return self.fc_out(out)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, x.size(1))
        return reconstructed_x, mu, logvar
    
    def size(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


