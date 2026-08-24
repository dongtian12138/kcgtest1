# GraspGenX 官方三指手最小正对照

一句话结论：官方 GraspGenX 在 RTX 5070 Ti 上完成一次 `robotiq_3f + banana.obj` 的 headless 正对照，模型只加载一次并输出 20 个有限的 4×4 抓取位姿及分数。

## 实际发生了什么

- 初始官方 `uv sync` 得到 Torch 2.6.0+cu124，无法执行 `sm_120`；经监督条件批准，只在现有隔离 `.venv` 内覆盖为官方 Torch 2.7.0+cu128 和 torchvision 0.22.0+cu128。
- 覆盖后 Torch、torchvision 均来自 GraspGenX `.venv`，CUDA 12.8，编译架构包含 `sm_120`；实际 GPU 矩阵运算成功。
- checkpoint 和 `robotiq_3f` 描述器从官方 Hugging Face 固定 revision 物化，真实 `banana.obj` 从 NVlabs 固定提交取得；所有 LFS 内容 SHA 与官方元数据一致。
- 模型加载一次，单次推理返回 20 个候选；没有再运行第二次模型或调参。

## 关键数据

- 模型加载：1.5961 s
- 推理：0.5224 s
- 探针总时间：2.3608 s
- 返回候选：20
- 分数范围：0.186690～0.986697
- GPU 峰值 allocated：616,143,360 B（约 587.6 MiB）
- GPU 峰值 reserved：673,185,792 B（约 642.0 MiB）
- 位姿有限：是
- 旋转正交最大误差：`3.94e-7`
- 模型加载次数：1

## 固定来源

- GraspGenX commit：`b9429097728cb1c430dd78b92edf17ba318aad03`
- checkpoint revision：`7c834043c11a11417e31d6d5ea9355801e40a2c1`
- checkpoint release tree：`ca884fbe760d41d9ca57cd3a6757d65105a58204`
- gripper descriptions revision：`19a03c00d19aeaf052d0f6801f0041982d676e8a`
- `robotiq_3f` tree：`4424cf709995cd1c980c01388cef9c33fc7cabec`
- `banana.obj` SHA-256：`701947c1f376efdd82de92758f80ca61e823301848efbfd9d645a62053055494`

## 证据边界

这是官方模型的静态 headless 正对照，只证明模型、权重、官方三指手描述和官方样例网格已经在本机 GPU 上连通，并能生成 20 个带分数的位姿。它没有检查连接器允许面、手指闭合路径、桌面碰撞、任务载荷余量、Isaac 抬升或硬件结果，不能写成 CARTS 动态成功。
