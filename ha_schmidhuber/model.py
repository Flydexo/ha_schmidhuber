import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.nn.modules.rnn import LSTM
        
class PrintShape(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        print("hey", x.shape)
        return x

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.mu = nn.Linear(1024, 32)
        self.log_sigma = nn.Linear(1024, 32)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        z = mu + torch.exp(log_sigma) * torch.randn_like(mu)
        return z, mu, log_sigma


class Fit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.unsqueeze(-1).unsqueeze(-1)
        
class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2),
            nn.ReLU(),
        )
        self.dense = Dense()
        self.decoder = nn.Sequential(
            nn.Linear(32, 1024),
            Fit(),
            nn.ConvTranspose2d(1024, 128, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2),
            nn.Sigmoid(),
        )

    def encode(self, x):
        # returns (z, mu, log_sigma)
        return self.dense(self.conv(x))

    def forward(self, x):
        z, mu, log_sigma = self.encode(x)
        x_recon = self.decoder(z)
        kl = -0.5 * (1 + 2 * log_sigma - mu.pow(2) - (2 * log_sigma).exp()).sum(-1).mean()
        return x_recon, kl

class MDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.rnn.hidden_size
        self.gaussians = cfg.rnn.num_mix
        self.z_dim = cfg.rnn.z_dim
        self.temp = cfg.rnn.temp
        self.layer = nn.Sequential(nn.Linear(h, h), nn.ReLU())
        self.probs_layer = nn.Linear(h, self.gaussians)
        self.means = nn.Linear(h, self.gaussians * self.z_dim)
        self.stds = nn.Linear(h, self.gaussians * self.z_dim)

    def forward(self, x):
        # fix #5: return distribution params for NLL loss, not a sampled point
        # x.shape = (B, 256)
        x = self.layer(x)
        pi = F.softmax(self.probs_layer(x) / self.temp, dim=-1)             # (B, 5)
        mu = self.means(x).view(-1, self.gaussians, self.z_dim)             # (B, 5, 32)
        sigma = torch.exp(self.stds(x)).view(-1, self.gaussians, self.z_dim) # (B, 5, 32)
        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        # fix #6: correct mixture sampling — pick one component, then sample from it
        # pi: (B, 5), mu/sigma: (B, 5, 32)
        k = torch.multinomial(pi, num_samples=1).squeeze(-1)                # (B,) — hard component draw
        B = mu.shape[0]
        mu_k = mu[torch.arange(B), k]                                       # (B, 32)
        sigma_k = sigma[torch.arange(B), k] * self.temp                     # (B, 32) — temperature scales uncertainty
        return Normal(mu_k, sigma_k).sample()                               # (B, 32)

    @staticmethod
    def loss(pi, mu, sigma, target):
        # NLL of target under the mixture: -log Σ_k π_k · N(target; μ_k, σ_k)
        # target: (B, 32)
        log_pi = torch.log(pi + 1e-8)                                       # (B, 5)
        log_prob = Normal(mu, sigma).log_prob(target.unsqueeze(1))          # (B, 5, 32)
        return -torch.logsumexp(log_pi + log_prob.sum(-1), dim=-1).mean()   # scalar
    

class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(cfg.rnn.z_dim + cfg.rnn.action_dim, cfg.rnn.hidden_size)
        self.mdn = MDN(cfg)

    def forward(self, z, a, hidden=None):
        # z: (T, z_dim) or (1, z_dim), a: (T, action_dim)
        # unsqueeze(1) adds batch dim → (T, 1, 35); LSTM treats dim 0 as seq_len
        x = torch.cat([z, a], dim=-1).unsqueeze(1)   # (T, 1, 35)
        output, hidden = self.lstm(x, hidden)         # output: (T, 1, 256)
        pi, mu, sigma = self.mdn(output.squeeze(1))   # (T, 256) → (T, ...)
        return pi, mu, sigma, hidden
        