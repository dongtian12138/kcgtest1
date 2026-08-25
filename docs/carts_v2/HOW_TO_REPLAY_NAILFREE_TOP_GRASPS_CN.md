# 如何复现 NF-HPGS 离线候选

## 当前可安全复现的范围

以下命令只复现两对象的高度投影、三指接触/闭合、12 N 任务排序和离散 bounded IK。它们不会启动真实硬件，也不会启动 Isaac 连接器抓取。

当前不提供完整抓取启动命令，因为无指甲 PhysX 碰撞资产尚未接受、完整七轴路径碰撞尚未闭合。绕过这两道门会把离线候选误当成动态安全候选。

## 环境和身份检查

```bash
cd /home/noob/WorkPlace/kcgtest1
git switch carts-grasp-nailfree-heightprojected-20260825
git status --short
sha256sum src/kcg_connector/config/carts_nailfree_height_projected.yaml
sha256sum artifacts/carts_v2/full_palm_search/proposals/current_d38999_26kj61sn_public_spec.json
sha256sum artifacts/carts_v2/full_palm_search/proposals/te_deutsch_d38999_26fj35pn_step.json
```

应核对的配置 SHA-256 为：

`87fdda414ccd8571526e7dae9e379880c6122762ec623b9adc153e4b77cc3350`

对象 A/B 提案 SHA-256 分别为：

- `aed02e498d476a79dd7e62c24e2edb6931f33c0862f31f97d298713acc4297d4`
- `0086d1be25fb3f74ef292a43676da5900c0bab881923e38eddcfcedf235caeea`

## 在新目录复现对象 A

完整 91 角级联耗时较长；使用新输出目录，避免覆盖保留证据。

```bash
cd /home/noob/WorkPlace/kcgtest1
export PYTHONPATH="$PWD/src/kcg_connector"
REPLAY_A_DIR="$(mktemp -d /tmp/nfhpgs_A_XXXXXX)"
third_party/GraspGenX/.venv/bin/python scripts/carts_v2/run_graspgenx_offline.py \
  --repository-root "$PWD" \
  --config "$PWD/src/kcg_connector/config/carts_nailfree_height_projected.yaml" \
  --baseline-config "$PWD/src/kcg_connector/config/carts_graspgenx_route1.yaml" \
  --integration-manifest "$PWD/artifacts/carts_v2/full_palm_search/SEARCH_MANIFEST.json" \
  --object-manifest "$PWD/artifacts/carts_v2/graspgenx/objects/object_manifest.json" \
  --proposal "$PWD/artifacts/carts_v2/full_palm_search/proposals/current_d38999_26kj61sn_public_spec.json" \
  --object-id current_d38999_26kj61sn_public_spec \
  --output-dir "$REPLAY_A_DIR" \
  --skip-coverage-render
```

## 在新目录复现对象 B

```bash
cd /home/noob/WorkPlace/kcgtest1
export PYTHONPATH="$PWD/src/kcg_connector"
REPLAY_B_DIR="$(mktemp -d /tmp/nfhpgs_B_XXXXXX)"
third_party/GraspGenX/.venv/bin/python scripts/carts_v2/run_graspgenx_offline.py \
  --repository-root "$PWD" \
  --config "$PWD/src/kcg_connector/config/carts_nailfree_height_projected.yaml" \
  --baseline-config "$PWD/src/kcg_connector/config/carts_graspgenx_route1.yaml" \
  --integration-manifest "$PWD/artifacts/carts_v2/full_palm_search/SEARCH_MANIFEST.json" \
  --object-manifest "$PWD/artifacts/carts_v2/graspgenx/objects/object_manifest.json" \
  --proposal "$PWD/artifacts/carts_v2/full_palm_search/proposals/te_deutsch_d38999_26fj35pn_step.json" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --output-dir "$REPLAY_B_DIR" \
  --skip-coverage-render
```

## 复现离散 bounded IK

以下示例读取已保留的离线结果，把新审计写入 `/tmp`：

```bash
cd /home/noob/WorkPlace/kcgtest1
export PYTHONPATH="$PWD/src/kcg_connector"
third_party/GraspGenX/.venv/bin/python \
  artifacts/carts_v2/nailfree_height_projected/offline_kinematic_routes/audit_bounded_ik.py \
  --repository-root "$PWD" \
  --config "$PWD/src/kcg_connector/config/carts_nailfree_height_projected.yaml" \
  --object-id current_d38999_26kj61sn_public_spec \
  --offline-result "$PWD/artifacts/carts_v2/nailfree_height_projected/offline_A/result.json" \
  --output /tmp/nfhpgs_object_A_bounded_ik.json

third_party/GraspGenX/.venv/bin/python \
  artifacts/carts_v2/nailfree_height_projected/offline_kinematic_routes/audit_bounded_ik.py \
  --repository-root "$PWD" \
  --config "$PWD/src/kcg_connector/config/carts_nailfree_height_projected.yaml" \
  --object-id te_deutsch_d38999_26fj35pn_step \
  --offline-result "$PWD/artifacts/carts_v2/nailfree_height_projected/offline_B/result.json" \
  --output /tmp/nfhpgs_object_B_bounded_ik.json
```

预期是 A 为 3/3、B 为 1/1，但这只表示离散位姿有逆解。

## 何时才允许复现 Isaac 抓取

必须先同时满足：

1. 三根无指甲末节的生产碰撞资产由可追溯闭合外壳生成，并通过可见/碰撞叠加和删除区零占用审计；
2. 离线网格、Isaac 碰撞网格、质量/质心/惯量和关节状态身份一致；
3. Top 候选完成七轴机械臂接近、逐指闭合和携带连接器抬升 50 mm 的完整路径碰撞检查；
4. Isaac 预检查失败关闭，在线控制继续只读关节位置、速度、误差、等效受力和时间；
5. `hardware_authorized=false` 保持。

满足后仍应先执行对象 A Top-3；只有 A 达到三指允许接触、离桌、50 mm、2 s、无未授权穿透，才冻结参数原样运行 B。
