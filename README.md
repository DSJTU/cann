# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。当前分支在单个 MIX kernel 内完成 Cube 得分、行最大值与最终归约；小型和窄矩阵使用 Vector。短 K 扩大 Cube 覆盖范围，按 batch/M/N 分工，一次请求最多覆盖 1024 列。长 K 将最多 256 列窗口的 128 宽 K 面板打包成连续 ND，一次 IterateBatch 算完后逐元素做 TwoSum 补偿，再取行最大值。Matmul 批次数补零到 2 的幂，避开 L1 批次拆分丢尾项，归约只读取有效 K 面板。提交源码为 [solution/kernel.asc](solution/kernel.asc)。

本分支仍在开发；当前源码不能继承此前版本的通过记录。`main` 上完成设备回归并经用户确认提升的版本仍是 `54cc8bd`。当前设备检查与赛事状态只记在本地 `.private`。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [历史记录](docs/README.md)
- [Ascend C 开发要点](docs/lessons.md)
- [实验目录说明](solution/experiments/README.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。日常迭代运行相关本地检查；关键检查点先设备回归，再由用户跑分。

官方 CPU Twin 本机运行与 GDB 调试见 [CPU Twin 入口](solution/tests/cpu_twin/README.md)。
