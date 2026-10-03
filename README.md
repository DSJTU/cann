# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。当前版本在单个 MIX kernel 内完成 Cube 得分、行最大值与最终归约；小型和窄矩阵使用 Vector。`54cc8bd` 已完成目标设备检查及赛事 15/15 验证，用户确认性能提升。提交源码为 [solution/kernel.asc](solution/kernel.asc)。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [当前检查点与历史记录](docs/README.md)
- [Ascend C 开发要点](docs/lessons.md)
- [实验目录说明](solution/experiments/README.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。日常迭代运行相关本地检查；关键检查点先设备回归，再由用户跑分。
