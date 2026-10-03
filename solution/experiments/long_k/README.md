# 长 K Cube 设计实验

本目录保留历史两次启动的设计探针；赛事要求每次迭代一个 kernel，因此探针不能用于提交。当前融合实现的验收使用 [单次启动检查点](../single_kernel/README.md)。

从已验证的 `fc8c8fc` kernel 和测试接口生成隔离实验，保留 ABI、图捕获资源管理和消费者同步。生成版本强制走 Cube，不能直接作为提交文件。实际候选位于 [主实现](../../kernel.asc)。

```bash
python3 solution/experiments/long_k/prepare.py /tmp/long-k-probes
python3 solution/experiments/long_k/prepare.py /tmp/long-k-grid --grid
python3 solution/experiments/long_k/prepare.py /tmp/long-k-partitions --partitioned
python3 /tmp/long-k-grid/k128/solution/tests/test_cpu.py --suite extended
```

三个计算版本使用 64×64 得分块：完整 K 的原生累加、512-K 补偿、128-K 补偿。默认沿 M 分工，`--grid` 对每个 M/N tile 分配独立任务，`--partitioned` 按可用核数限制 N 分区，每个分区可串行处理多个 N tile。各分区独立写行最大值，Finish 先逐行合并 N 分区，再用补偿树求和 M；复用 low 缓冲做 N 合并输入，自有 UB 大小不增长。

生成器读取固定 Git 提交，后续主实现变化不会静默改变已记录实验。需要在包含该提交的 Git checkout 中运行。各版本的源码 SHA256 记入生成目录的 manifest.json。拒绝复用非空目录。

`data.py` 支持 `precision`、`benchmark` 和 `intra-chunk`。精度集合包含 K=256/392/1024/8192、M/N 尾部、相邻 batch、两类型、四布局、全负、跨 K 抵消、不同量级、接近最大值，以及抵消发生在单个 512-K 分段内的输入。黄金值由实际量化存储值独立以 FP64 计算。CPU Matmul 的 FP64 模型不能发现设备 Cube 的舍入错误。

目标设备加载 CANN SDK 后，从生成目录运行 `checkpoint.py`。它构建基线及三种版本，保存数值失败与原始性能记录；`design_checkpoint_completed` 只表示实验完成，不表示全部版本精度通过。每组性能数据预热 2 次、测量 10 次；Cube 区间从 Scores 开始到 Finish 结束，包含两 kernel 的间隔。

`candidate_check.py` 用于隔离的实际候选检查点：目录需包含 `candidate/solution`、`baseline/solution` 及源码哈希的 `candidate-manifest.json`。它检查实际分派的 72 个 long-cube 输入、24 个大型短 K 归约输入和 32 个小型输入，以及重复、捕获和独立动态库调用，再同轮比较两类固定性能输入。脚本中的输入路径对应本次授权设备实验，迁移环境时需先恢复同一输入并核对哈希。

实验结论与实际候选验证见 [长 K 记录](../../../docs/long-k-performance.md)。
