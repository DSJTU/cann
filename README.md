# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局与 FP32 输出。提交入口为 [solution/kernel.asc](solution/kernel.asc)，每次调用恰好启动一个 kernel。

当前分支重建为原生完整 K 的 Cube 架构：直接读取原始 ND 输入，按 batch/M/N 分配任务，立即归约每个得分块，最后在同一个 MIX kernel 中合并行最大值。短/中 K 的宽矩阵使用较宽 N 块；长 K 且任务不足时减小 M/N 块以增加并行度。小矩阵使用缓存输入的 Vector 路径，M=N=1 使用专用点积。

这次重建移除了旧版 K 面板打包和逐面板补偿。完整 K 的硬件 FP32 累加仍存在已知的强抵消精度限制；CPU 数学模型不能证明这类输入通过。当前分支尚待赛事评测，设备检查与成绩历史记录在本地 `.private`。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 接口与同步要点](docs/lessons.md)
- [官方 CPU Twin 本机调试](solution/tests/cpu_twin/README.md)
- [历史实验](solution/experiments/README.md)

本地检查：

```bash
python3 solution/tests/test_cpu.py
python3 solution/tests/test_finish.py
```

关键检查点先验证目标 ASC 编译、必要的真机调用模式与实际 kernel 启动数量，再由用户跑赛事分数。
