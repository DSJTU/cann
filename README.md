# BatchMatmulMaxSum

Ascend C 比赛算子。提交 [solution/kernel.asc](solution/kernel.asc) 全部内容，支持 FP16/BF16、四种布局和 FP32 输出，每次调用只启动一个 kernel。

短 K 使用直接 Cube 指令；窄矩阵长 K 在同一个 kernel 内重排 NZ，并在 L1 复用完整 A 面板。其余矩阵使用 SDK Matmul 的异步 N 窗口。Cube 完成 K 点积后，Vector 取行最大值并归约；很小的矩阵和单点积使用 Vector 专用路径。

- [赛题契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [接口与同步要点](docs/lessons.md)

本地检查：`python3 solution/tests/test_cpu.py`。它不能替代目标编译、真机 profile 和赛事成绩。

个人环境、最佳成绩证据与简要优化经验放在不提交的 `.private/`；只保留最佳赛事版本。生成物统一放 `.private/runtime/`，失败实验归纳后清理。
