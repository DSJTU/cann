# BatchMatmulMaxSum

Ascend C 实现，支持 FP16/BF16 输入、四种存储布局与 FP32 输出。提交入口为 [solution/kernel.asc](solution/kernel.asc)，每次调用恰好启动一个 kernel。

Cube 直接读取原始 ND 输入，按 batch/M/N 分配任务，逐块归约得分，最后在同一个 MIX kernel 中合并行最大值。小矩阵使用缓存输入的 Vector 路径，M=N=1 使用专用点积。

完整 K 的硬件 FP32 累加有强抵消精度限制；CPU 数学模型不模拟这类硬件舍入。实现结构和验证范围见构建说明。

- [算子契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [Ascend C 接口与同步要点](docs/lessons.md)
- [官方 CPU Twin 本机调试](solution/tests/cpu_twin/README.md)

本地检查：

```bash
python3 solution/tests/test_cpu.py
python3 solution/tests/test_finish.py
```

共用目录只保存算子契约、提交实现、构建工具和可复用测试。个人环境、参考代码、实验、成绩和开发记录放在 Git 忽略的 `.private/`；提交前检查暂存区内容。
