# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。此前的 Vector 正确性基线和性能优化版本均通过赛事 15/15，见 [验证记录](docs/validation.md) 与 [Vector 性能记录](docs/performance.md)。当前加入大型短 K 的 Cube 路径，验证与计时单独记录在 [Cube 性能记录](docs/cube-performance.md)。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 开发要点](docs/lessons.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。
