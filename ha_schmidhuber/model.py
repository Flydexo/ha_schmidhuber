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

    @staticmethod
    def kl_divergence(mu, log_sigma):
        # KL(N(mu, sigma^2) || N(0,1)) summed over latent dims, averaged over the batch.
        # log_sigma is log-STD (Dense samples with std = exp(log_sigma)), so variance is
        # exp(2*log_sigma). This is 2x the textbook KL -- the global 0.5 is dropped to match
        # the sum-reduced MSE recon (also 2x a unit-variance Gaussian NLL), keeping the recon:KL
        # scale (and thus beta / the free-bits floor) consistent.
        var = torch.exp(2 * log_sigma)
        return (mu.pow(2) + var - 2 * log_sigma - 1).sum(-1).mean()

    def forward(self, x):
        # x.shape = B * C * H * W
        z, mu, log_sigma = self.encode(x)
        # (z,mu,log_sigma).shape = B * 32
        x_recon = self.decoder(z)
        kl = self.kl_divergence(mu, log_sigma)
        return x_recon, kl

class MDN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.rnn.hidden_size
        self.gaussians = cfg.rnn.num_mix
        self.z_dim = cfg.rnn.z_dim
        self.temp = cfg.rnn.temp
        #self.layer = nn.Sequential(nn.Linear(h, h), nn.ReLU())
        self.probs_layer = nn.Linear(h, self.gaussians)
        self.means = nn.Linear(h, self.gaussians * self.z_dim)
        self.stds = nn.Linear(h, self.gaussians * self.z_dim)

    def forward(self, x):
        # fix #5: return distribution params for NLL loss, not a sampled point
        # x.shape = (B, 256)
        #x = self.layer(x)
        temp = self.temp if not(self.training) else 1
        pi = F.softmax(self.probs_layer(x) / temp, dim=-1)             # (B, 5)
        mu = self.means(x).view(-1, self.gaussians, self.z_dim)             # (B, 5, 32)
        sigma = torch.exp(self.stds(x)).view(-1, self.gaussians, self.z_dim) # (B, 5, 32)
        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        # fix #6: correct mixture sampling — pick one component, then sample from it
        # pi: (B, 5), mu/sigma: (B, 5, 32)
        k = torch.multinomial(pi, num_samples=1).squeeze(-1)                # (B,) — hard component draw
        B = mu.shape[0]
        mu_k = mu[torch.arange(B), k]                                       # (B, 32)
        sigma_k = sigma[torch.arange(B), k]                    # (B, 32) — temperature scales uncertainty
        return Normal(mu_k, sigma_k).sample()                               # (B, 32)

    @staticmethod
    def loss(pi, mu, sigma, target, mask=None):
        # Works for any prefix shape: (B, 32) or (B, T, 32)
        log_pi = torch.log(pi + 1e-8)                                       # (..., K)
        log_prob = Normal(mu, sigma).log_prob(target.unsqueeze(-2))         # (..., K, 32)
        nll = -torch.logsumexp(log_pi + log_prob.sum(-1), dim=-1)           # (...)
        return nll[mask].mean() if mask is not None else nll.mean()
    

class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.lstm = LSTM(cfg.rnn.z_dim + cfg.rnn.action_dim, cfg.rnn.hidden_size, batch_first=True)
        self.mdn = MDN(cfg)

    def forward(self, z, a, hidden=None):
        # z: (B, T, z_dim) or (T, z_dim) for single episode
        # a: (B, T, action_dim) or (T, action_dim)
        x = torch.cat([z, a], dim=-1)                 # (B, T, 35) or (T, 35)
        if x.dim() == 2:
            x, squeeze = x.unsqueeze(0), True
        else:
            squeeze = False
        output, hidden = self.lstm(x, hidden)          # (B, T, 256)
        B, T, H = output.shape
        pi, mu, sigma = self.mdn(output.reshape(B * T, H))
        pi    = pi.view(B, T, self.mdn.gaussians)
        mu    = mu.view(B, T, self.mdn.gaussians, self.mdn.z_dim)
        sigma = sigma.view(B, T, self.mdn.gaussians, self.mdn.z_dim)
        if squeeze:
            pi, mu, sigma, output = pi.squeeze(0), mu.squeeze(0), sigma.squeeze(0), output.squeeze(0)
        return pi, mu, sigma, hidden, output
        