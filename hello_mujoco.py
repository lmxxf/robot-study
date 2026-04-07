import mujoco
import mediapy

# MuJoCo 用 XML 描述机器人模型（叫 MJCF 格式）
# 这里做一个钟摆：铰链在上，杆子挂下来，末端一个球
XML = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <body pos="0 0 2">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" rgba="0.8 0.8 0.8 1"/>
      <body pos="0 0 -0.5">
        <geom type="sphere" size="0.08" rgba="1 0 0 1" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

# 给钟摆一个初始角度，松手让它摆
data.qpos[0] = 1.5  # 1.5 弧度（约 86 度）

# 渲染器
renderer = mujoco.Renderer(model, width=640, height=480)

# 跑 2000 步物理仿真（默认 timestep=0.002s，共 4 秒）
frames = []
for i in range(2000):
    mujoco.mj_step(model, data)
    if i % 10 == 0:  # 每 10 步存一帧，30fps 的视频
        renderer.update_scene(data)
        frames.append(renderer.render().copy())

# 存成 mp4
mediapy.write_video("hello_mujoco.mp4", frames, fps=30)
print(f"视频已保存，共 {len(frames)} 帧")
