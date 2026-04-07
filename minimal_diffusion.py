"""
最小扩散模型：用 PyTorch 从零实现 DDPM，学习生成 2D 螺旋线上的点。

核心逻辑和 Push-T 的 Diffusion Policy 完全一致，只是：
- 数据从 32 维轨迹换成了 2 维的 (x, y) 点
- 网络从 1D U-Net 换成了 3 层 MLP（2D 点太简单，MLP 就够）

运行：python minimal_diffusion.py
输出：spiral_diffusion_result.png（左边原始数据，右边生成结果）
耗时：CPU 约 1 分钟
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ============================================================
# 超参数
# ============================================================
T = 300          # 加噪总步数（泼多少遍噪声）
EPOCHS = 5000    # 训练轮数
BATCH = 512      # 每批多少个点
LR = 1e-3        # 学习率
N_POINTS = 2000  # 螺旋线上的点数（训练数据量）
N_GEN = 1000     # 推理时生成多少个点

# ============================================================
# 第一段：准备数据和工具
# ============================================================

# ---- 数据：生成螺旋线上的点 ----
# 螺旋线在 2D 平面上，每个点是 (x, y) 坐标
# 这就是我们的"训练数据分布"——模型要学会从噪声中生成这个分布的样本
def make_spiral(n_points=N_POINTS):
    theta = torch.linspace(0, 4 * math.pi, n_points) + torch.randn(n_points) * 0.1
    r = theta / (4 * math.pi)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack([x, y], dim=1)  # shape: (n_points, 2)


# ---- 噪声调度：预计算 alpha_bar ----
# beta 从 0.0001 线性增长到 0.02，共 T=300 步
# alpha_bar_t = 累乘 (1 - beta_1) * (1 - beta_2) * ... * (1 - beta_t)
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas                     # alpha_t = 1 - beta_t（纯定义）
alpha_bars = torch.cumprod(alphas, dim=0) # 累乘，核心就这一行


# ---- 神经网络：3 层 MLP ----
# 输入：加噪后的 2D 坐标 (2维) + 时间步嵌入 (64维) = 66 维
# 输出：预测的噪声 (2维)
# 没有用 U-Net，因为 2D 点太简单了，MLP 就够
class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Embedding(T, 64)   # 把整数时间步变成 64 维向量
        self.net = nn.Sequential(
            nn.Linear(2 + 64, 256), nn.ReLU(),   # 66 维输入 → 256
            nn.Linear(256, 256), nn.ReLU(),       # 256 → 256
            nn.Linear(256, 2),                    # 256 → 2 维输出（预测的噪声）
        )

    def forward(self, x_t, t):
        t_emb = self.time_embed(t)                              # (batch, 64)
        return self.net(torch.cat([x_t, t_emb], dim=1))         # (batch, 2)


# 从训练数据中随机采一批
def sample_batch(data, batch_size=BATCH):
    idx = torch.randint(0, len(data), (batch_size,))
    return data[idx]


# ============================================================
# 第二段：训练（出题 → 答题 → 对答案，循环 5000 次）
# ============================================================

data = make_spiral()
model = NoisePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print(f"开始训练：{EPOCHS} 轮，每轮 {BATCH} 个点，加噪 {T} 步")

for epoch in range(EPOCHS):
    x0 = sample_batch(data)                           # 1. 取一批螺旋线上的点
    t = torch.randint(0, T, (BATCH,))                 # 2. 随机选时间步（泼几遍）
    eps = torch.randn_like(x0)                         # 3. 掷骰子生成随机数

    # 4. 正向加噪：一步到位（出题）
    ab_t = alpha_bars[t].unsqueeze(1)                  # (batch, 1)
    xt = torch.sqrt(ab_t) * x0 + torch.sqrt(1 - ab_t) * eps

    # 5. 网络猜噪声（答题）
    eps_pred = model(xt, t)

    # 6. 对答案
    loss = F.mse_loss(eps_pred, eps)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f"  第 {epoch+1} 轮，Loss = {loss.item():.4f}")

print("训练完成！")


# ============================================================
# 第三段：推理（从纯随机数变出螺旋线）
# ============================================================

print(f"开始生成：从 {N_GEN} 个纯随机点出发，去噪 {T} 步")

model.eval()
with torch.no_grad():
    x = torch.randn(N_GEN, 2)  # 从纯噪声开始（1000 个随机的 (x,y) 点）

    for t_step in reversed(range(T)):
        t_batch = torch.full((N_GEN,), t_step, dtype=torch.long)

        # 网络猜噪声
        eps_pred = model(x, t_batch)

        # 减掉猜的噪声，缩放
        x = (1 / torch.sqrt(alphas[t_step])) * (
            x - betas[t_step] / torch.sqrt(1 - alpha_bars[t_step]) * eps_pred
        )

        # 加一点随机扰动（最后一步不加）
        if t_step > 0:
            x = x + torch.sqrt(betas[t_step]) * torch.randn_like(x)

generated = x.numpy()

print("生成完成！")


# ============================================================
# 画图：左边原始数据，右边生成结果
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：原始螺旋线
original = data.numpy()
ax1.scatter(original[:, 0], original[:, 1], s=3, alpha=0.5, c='#2a7fff')
ax1.set_title('Original Spiral (Training Data)', fontsize=14)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.2, 1.2)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# 右图：生成的螺旋线
ax2.scatter(generated[:, 0], generated[:, 1], s=3, alpha=0.5, c='#ff6644')
ax2.set_title('Generated Spiral (From Pure Noise)', fontsize=14)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spiral_diffusion_result.png', dpi=150, bbox_inches='tight')
print("结果已保存到 spiral_diffusion_result.png")
