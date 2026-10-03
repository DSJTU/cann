# 实验索引

提交使用 [kernel.asc](../kernel.asc)。实验源码与检查点脚本按阶段保留，便于复现精度反例、设计选择和性能结论。

| 目录 | 用途 | 状态 |
| --- | --- | --- |
| [single_kernel/](single_kernel/README.md) | 当前融合实现的设备检查点 | 当前验收入口 |
| [long_k/](long_k/README.md) | 固定 `fc8c8fc` 的长 K 精度和任务分配探针 | 历史两次启动实验 |
| [cube/](cube/README.md) | 32×64 短 K Cube 原型 | 历史两次启动实验 |

历史探针保留原目录以维持固定提交生成器和构建路径。两次启动实验违反当前赛事规则；复现结论以各阶段源码哈希与输入为准。生成的 build、runs、lab-data、profile 和 Python 缓存属于忽略产物，源码及验收数据保留在 Git 中。
