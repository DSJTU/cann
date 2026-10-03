# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。此前的 Vector 正确性基线和性能优化版本均通过赛事 15/15，见 [验证记录](docs/validation.md) 与 [Vector 性能记录](docs/performance.md)。大型短 K 的 Cube 路径历史验证见 [Cube 性能记录](docs/cube-performance.md) 与 [Finish 检查点](docs/finish-performance.md)。原长 K 两次启动版本的设备实验见 [长 K 检查点](docs/long-k-performance.md)；赛事要求每次迭代恰好一次 kernel 启动，当前改为单 MIX kernel 完成 Cube 和最终归约，验证见 [单次启动修复](docs/single-kernel.md)。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 开发要点](docs/lessons.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。
