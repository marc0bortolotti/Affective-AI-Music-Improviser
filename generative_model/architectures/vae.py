import torch
import torch.nn as nn

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, vocab_size, embed_dim):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Output mean and log variance
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output must be in [0, 1] for binary cross-entropy
        )

        self.PARAMS = { 'input_dim': input_dim,
                        'hidden_dim': hidden_dim,
                        'latent_dim': latent_dim,
                        'vocab_size': vocab_size,
                        'embed_dim': embed_dim,
                    }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        h1 = self.encoder(x)
        mu, logvar = h1.chunk(2, dim=-1)  # Split into mean and log variance
        z = self.reparameterize(mu, logvar)  # Sample from latent space
        # Decode
        output = self.decoder(z)
        return output, mu, logvar

