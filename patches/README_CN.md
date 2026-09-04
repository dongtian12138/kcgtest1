# 外部依赖补丁

## SAM-6D

本工程没有复制 SAM-6D 源码、模型权重或运行缓存。当前视觉实验使用的上游版本为：

- 仓库：`https://github.com/JiehongLin/SAM-6D.git`
- 基准提交：`1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2`
- 本地补丁：`sam6d_runtime_determinism.patch`

在 SAM-6D 仓库根目录执行：

```bash
git checkout 1c2543b3b6faa1f1d81b3c7291f8b371d71e50c2
git apply /path/to/kcgtest1/patches/sam6d_runtime_determinism.patch
```

补丁只包含本工程实际使用的三类运行修正：固定 CAD 点采样和 CUDA 随机源、修正命令行浮点参数类型，以及兼容没有预置材质的 CAD 模型。模型权重和 Python/CUDA 环境仍需按 SAM-6D 上游说明安装。
