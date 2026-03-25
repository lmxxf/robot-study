# 开发日志

---

## 2026-03-25 环境搭建

### 做了什么

1. 在 `d2l_exp` 容器（`nvcr.io/nvidia/pytorch:25.11-py3`）里安装 LeRobot
   - 容器自带 Python 3.12.3，满足 LeRobot 0.5.0 要求
   - `pip install 'lerobot[aloha,pusht]'` 一把装好，含 MuJoCo 3.6.0、gym-aloha、gym-pusht
   - 安装过程升级了 huggingface-hub（0.36→1.7），导致旧项目 ctx-to-lora 报依赖冲突，但该库已自动卸载，无影响

2. 设置无头渲染：`export MUJOCO_GL=egl`（容器无显示器）

3. 准备跑第一个 demo：Push-T（Diffusion Policy 推 T 形积木）

### 为什么做

- 目标不是搞机器人工程，是**理解 VLA 是什么**，跑仿真看个直观感受，然后写公众号
- 选 LeRobot 框架：HuggingFace 出品，生态好，从仿真到真机一条龙
- 选 Docker 而不是本地 venv：d2l_exp 容器有 GPU + PyTorch，避免污染宿主机
- Push-T 是最简单的入门任务，验证环境能跑通

### 命令记录

```bash
# 进容器
docker exec -it d2l_exp bash

# 安装 LeRobot
pip install 'lerobot[aloha,pusht]'

# 验证
python3 -c "import lerobot; print(lerobot.__version__)"   # 0.5.0
python3 -c "import mujoco; print(mujoco.__version__)"     # 3.6.0

# 无头渲染（容器里每次进来都要设）
export MUJOCO_GL=egl

# 跑 Push-T demo（Diffusion Policy 预训练模型，10 局仿真）
# 注意：lerobot 0.5.0 入口改了，不是 lerobot.scripts.eval，是 lerobot.scripts.lerobot_eval
python3 -m lerobot.scripts.lerobot_eval --policy.path=/workspace/models/diffusion_pusht --env.type=pusht --env.task=PushT-v0 --eval.n_episodes=10 --eval.batch_size=10 --output_dir=outputs/eval/pusht_demo
```

### 踩坑记录

1. **容器网络**：容器 bridge 模式，pip 源能访问，但 huggingface.co 不通（被墙）
   - 解法：宿主机下载模型，拷进容器
   - `huggingface-cli download lerobot/diffusion_pusht --local-dir /tmp/diffusion_pusht`
   - `docker cp /tmp/diffusion_pusht d2l_exp:/workspace/models/diffusion_pusht`

2. **GPU 未识别**：容器启动时 NVML 初始化失败，fallback 到 CPU
   - 解法：`docker restart d2l_exp` 后恢复（GPU 重新挂载）

3. **opencv 缺 libxcb**：`ImportError: libxcb.so.1: cannot open shared object file`
   - 解法：`pip install opencv-python-headless --force-reinstall`（headless 版不需要 xcb）
   - 注意降回兼容版本：`pip install 'numpy>=2.0.0,<2.3.0' 'opencv-python-headless>=4.9.0,<4.13.0'`

4. **huggingface-hub 版本冲突**：lerobot 装了 1.7.2，transformers 要 <1.0
   - 解法：`pip install transformers -U`

5. **模型格式需要迁移**：预训练模型缺 `policy_preprocessor.json`
   - 解法（待执行）：`python3 -m lerobot.processor.migrate_policy_normalization --pretrained-path /workspace/models/diffusion_pusht`
