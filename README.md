# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。当前分支在单个 MIX kernel 内完成 Cube 得分、行最大值与最终归约；小型和窄矩阵使用 Vector。短 K 一次请求最多覆盖 1024 列。长 K 把每个 64×64 块的 128 宽面板打包成连续 ND，一次 IterateBatch 算完后再补偿。提交源码为 [solution/kernel.asc](solution/kernel.asc)。

本分支仍在开发。最近一次赛事是 15/15、错误占比 0%，逐点耗时只记在本地 `.private`。`main` 上完成设备回归并经用户确认提升的版本仍是 `54cc8bd`。本分支还没有对应的设备回归，也还不是新的合并基线。

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
