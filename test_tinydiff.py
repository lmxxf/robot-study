"""
直接用 tiny-diffusion 的代码跑 moons 数据集，验证能跑通。
然后换成螺旋线。
"""
import sys
sys.path.insert(0, '/tmp/tiny-diffusion')

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from positional_embeddings import PositionalEmbedding

# ---- 直接复制 tiny-diffusion 的 Block 和 MLP ----
class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.ff(x))

class MLP(nn.Module):
    def __init__(self, hidden_size=128, hidden_layers=3, emb_size=128,
                 time_emb="sinusoidal", input_emb="sinusoidal"):
        super().__init__()
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, x, t):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        t_emb = self.time_mlp(t)
        x = torch.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x

# ---- 直接复制 tiny-diffusion 的 NoiseScheduler ----
class NoiseScheduler:
    def __init__(self, num_timesteps=50, beta_start=0.0001, beta_end=0.02, beta_schedule="linear"):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(1 / self.alphas_cumprod - 1)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        s1 = self.sqrt_inv_alphas_cumprod[t].reshape(-1, 1)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].reshape(-1, 1)
        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        s1 = self.posterior_mean_coef1[t].reshape(-1, 1)
        s2 = self.posterior_mean_coef2[t].reshape(-1, 1)
        return s1 * x_0 + s2 * x_t

    def get_variance(self, t):
        if t == 0:
            return 0
        return (self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])).clip(1e-20)

    def step(self, model_output, timestep, sample):
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise
        return pred_prev_sample + variance

    def add_noise(self, x_start, x_noise, timesteps):
        s1 = self.sqrt_alphas_cumprod[timesteps].reshape(-1, 1)
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps].reshape(-1, 1)
        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps

# ---- 螺旋线数据 ----
def make_spiral(n=8000):
    theta = torch.linspace(0, 4 * math.pi, n) + torch.randn(n) * 0.1
    r = theta / (4 * math.pi)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    data = torch.stack([x, y], dim=1) * 4  # 放大到 [-4, 4]
    return TensorDataset(data)

# ---- 训练 ----
dataset = make_spiral()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
model = MLP()
noise_scheduler = NoiseScheduler(num_timesteps=50)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training...")
for epoch in range(200):
    model.train()
    for step, (batch,) in enumerate(dataloader):
        noise = torch.randn(batch.shape)
        timesteps = torch.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
        noisy = noise_scheduler.add_noise(batch, noise, timesteps)
        noise_pred = model(noisy, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch+1}, Loss = {loss.item():.4f}")

# ---- 生成 ----
print("Generating...")
model.eval()
sample = torch.randn(1000, 2)
timesteps = list(range(len(noise_scheduler)))[::-1]
for t in timesteps:
    t_batch = torch.from_numpy(np.repeat(t, 1000)).long()
    with torch.no_grad():
        residual = model(sample, t_batch)
    sample = noise_scheduler.step(residual, t_batch[0], sample)
generated = sample.numpy()

# ---- 画图 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
original = dataset.tensors[0].numpy()
ax1.scatter(original[:, 0], original[:, 1], s=3, alpha=0.5, c='#2a7fff')
ax1.set_title('Original Spiral', fontsize=14)
ax1.set_xlim(-6, 6); ax1.set_ylim(-6, 6); ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)
ax2.scatter(generated[:, 0], generated[:, 1], s=3, alpha=0.5, c='#ff6644')
ax2.set_title('Generated (From Noise)', fontsize=14)
ax2.set_xlim(-6, 6); ax2.set_ylim(-6, 6); ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('spiral_diffusion_result.png', dpi=150, bbox_inches='tight')
print("Done! → spiral_diffusion_result.png")
