"""
对照实验：不训练，直接用随机权重的网络去噪。
如果也出螺旋线 → 之前的结果是假的。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ---- 内联必要的类 ----
class SinusoidalEmbedding(nn.Module):
    def __init__(self, size=128, scale=1.0):
        super().__init__()
        self.size = size
        self.scale = scale
    def forward(self, x):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0])) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)

class Block(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()
    def forward(self, x):
        return x + self.act(self.ff(x))

class NoisePredictor(nn.Module):
    def __init__(self, hidden=128, n_layers=3, emb_size=128):
        super().__init__()
        self.time_emb = SinusoidalEmbedding(emb_size, scale=1.0)
        self.input_emb_x = SinusoidalEmbedding(emb_size, scale=25.0)
        self.input_emb_y = SinusoidalEmbedding(emb_size, scale=25.0)
        layers = [nn.Linear(emb_size * 3, hidden), nn.GELU()]
        for _ in range(n_layers):
            layers.append(Block(hidden))
        layers.append(nn.Linear(hidden, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x_t, t):
        x_emb = self.input_emb_x(x_t[:, 0])
        y_emb = self.input_emb_y(x_t[:, 1])
        t_emb = self.time_emb(t)
        return self.net(torch.cat([x_emb, y_emb, t_emb], dim=-1))

class NoiseScheduler:
    def __init__(self, T=50):
        self.betas = torch.linspace(1e-4, 0.02, T, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)
        self.sqrt_inv_alpha_bars = torch.sqrt(1.0 / self.alpha_bars)
        self.sqrt_inv_alpha_bars_m1 = torch.sqrt(1.0 / self.alpha_bars - 1)
        self.post_coef1 = self.betas * torch.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars)
        self.post_coef2 = (1. - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1. - self.alpha_bars)

    def step(self, eps_pred, t, x_t):
        x0 = self.sqrt_inv_alpha_bars[t].reshape(-1,1) * x_t - self.sqrt_inv_alpha_bars_m1[t].reshape(-1,1) * eps_pred
        mean = self.post_coef1[t].reshape(-1,1) * x0 + self.post_coef2[t].reshape(-1,1) * x_t
        if t > 0:
            var = (self.betas[t] * (1.-self.alpha_bars_prev[t]) / (1.-self.alpha_bars[t])).clamp(1e-20)
            mean = mean + (var**0.5) * torch.randn_like(x_t)
        return mean

# ---- 不训练，直接用随机权重生成 ----
model = NoisePredictor()  # 随机权重
scheduler = NoiseScheduler(T=50)

model.eval()
with torch.no_grad():
    x = torch.randn(1000, 2)
    for t_step in reversed(range(50)):
        t_batch = torch.from_numpy(np.repeat(t_step, 1000)).long()
        eps_pred = model(x, t_batch)
        x = scheduler.step(eps_pred, t_batch[0], x)

generated = x.numpy()

# 画图
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(generated[:, 0], generated[:, 1], s=3, alpha=0.5, c='#ff6644')
ax.set_title('NO TRAINING - Random Weights', fontsize=14)
ax.set_xlim(-6, 6); ax.set_ylim(-6, 6); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('no_train_result.png', dpi=150, bbox_inches='tight')
print("Done → no_train_result.png")
