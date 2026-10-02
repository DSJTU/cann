# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局和 FP32 输出。当前赛事评测通过 15/15。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 开发要点](docs/lessons.md)

本地检查依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
```

设备编译与真机验证需要匹配的 CANN SDK 和 NPU。
