# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。正确性基线通过赛事评测 15/15，性能优化版本也通过本轮赛事评测 15/15；截图与逐项耗时见 [性能记录](docs/performance.md)。基线及性能优化版本的真机验证数据分别见 [验证记录](docs/validation.md) 和 [性能记录](docs/performance.md)。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 开发要点](docs/lessons.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。
