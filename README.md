# dsm-main: 双边匹配任务分配研究项目

本仓库用于研究任务-提供者双边匹配问题，主要包含基于强化学习的方法（DQN/PPO 变体）以及传统启发式方法（GA/ACO/PSO）。

核心目标是：在任务约束、预算、资源与服务方属性等条件下，学习或搜索出高质量匹配方案，并输出可复现实验结果与可视化数据。

## 1. 项目特点

- 多模型并存：`Basic DQN`、`No-Attention DQN`、`XTMQN`、`PPO`。
- 数据驱动：训练与评估由 JSON 场景文件驱动。
- 输出完整：保存模型权重、匹配点、损失曲线、目标值和汇总文件。
- 便于切换：可在同一训练入口快速切换不同网络实现。

## 2. 目录结构（重点）

```text
.
├─ dsm-main/
│  ├─ Nature/
│  │  ├─ doubleSideMatching.py   # 训练/运行主入口
│  │  ├─ basic_dqn.py            # 基础 DQN
│  │  ├─ no_attention_dqn.py     # 去 Attention 的 DQN
│  │  ├─ xtmqn.py                # XTMQN 变体（当前默认导入）
│  │  └─ PPO.py                  # PPO 变体
│  ├─ test_basic_dqn.py          # Basic DQN 快速冒烟测试
│  ├─ test_no_attention_dqn.py   # No-Attention DQN 快速测试
│  ├─ parameter_analysis.py      # 参数量分析脚本
│  ├─ 60_30/                     # 样例数据集目录（部分）
│  └─ data_*.json                # 场景数据
├─ 10_5/                         # 额外数据集目录（workspace 级）
├─ plots/
└─ visualizations/
```

## 3. 环境准备

建议 Python 版本：`3.9+`

安装基础依赖（至少）：

```bash
pip install torch numpy matplotlib
```

如果需要使用 Gurobi 相关功能（可选）：

```bash
pip install gurobipy
```

## 4. 快速开始

### 4.1 训练主流程

从 workspace 根目录执行：

```bash
python dsm-main/Nature/doubleSideMatching.py
```

当前脚本默认会读取 `doubleSideMatching.py` 中的 `datafile` 并执行：

- `main(2000, det)`

你可以在 `dsm-main/Nature/doubleSideMatching.py` 中修改：

- `datafile`：切换训练数据。
- `main(2000, det)`：调整训练轮数。
- `device = "cuda"`：无 GPU 环境可改为 `"cpu"`。

### 4.2 运行测试脚本

```bash
python dsm-main/test_basic_dqn.py
python dsm-main/test_no_attention_dqn.py
```

### 4.3 参数量分析

```bash
python dsm-main/parameter_analysis.py
```

## 5. 模型切换

在 `dsm-main/Nature/doubleSideMatching.py` 顶部，通过替换导入切换实现。

示例：

```python
# from PPO import network, point
from xtmqn import network, point
# from basic_dqn import network, point
# from no_attention_dqn import network, point
```

将 `from xtmqn import network, point` 替换为目标实现即可。

## 6. 输入与输出说明

### 6.1 输入数据

数据为 JSON 文件（如 `data_60_30.json`、`60_30/data_60_30_13096.json`），常见字段包括：

- `taskNum`, `providerNum`
- `taskTime`, `providerPrice`, `providerRep`
- `taskdeadlines`, `taskbudgets`, `taskResources`
- `edges`, `andor` 等

### 6.2 训练输出（常见）

- 模型权重：`model_{task}_{provider}_ppo.pth`
- 搜索点：`points_{task}_{provider}_ppo.pkl`
- 损失与目标序列：`losses.pkl`, `objvs.pkl`, `episode_rewards.pkl`
- 绘图中间数据：`draw.json`, `draw_attnet.json`
- 可视化导出：`visualizations/` 下的 `csv/json/png`

## 7. 常见问题

### 7.1 没有 GPU / CUDA 报错

将 `dsm-main/Nature/doubleSideMatching.py` 中：

```python
device = "cuda"
```

改为：

```python
device = "cpu"
```

### 7.2 导入失败（找不到 Nature 内模块）

优先从 workspace 根目录运行命令，避免相对路径与 `sys.path` 差异导致导入错误。

### 7.3 路径分隔符问题（Windows/跨平台）

建议在自定义脚本中优先使用 `/` 或 `os.path.join`，避免硬编码 `\\`。

## 8. 复现建议

- 固定随机种子（项目中已有 `torch.manual_seed` 和 `np.random.seed` 示例）。
- 记录数据文件、模型实现、训练轮数与关键超参数。
- 对比不同实现时，保持相同数据与预算条件。

## 9. License

当前仓库未显式声明 License。如需开源发布，建议补充 `LICENSE` 文件并在 README 中注明。
