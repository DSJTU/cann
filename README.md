# BatchMatmulMaxSum

Ascend C 比赛算子。提交 [solution/kernel.asc](solution/kernel.asc) 全部内容，支持 FP16/BF16、四种布局和 FP32 输出，每次调用只启动一个 kernel。

Cube 沿完整 K 计算，在 N 窗口内异步输出独立 GM 槽，Vector 同时取行最大值；需要跨核合并时在同一个 MIX kernel 内完成。小矩阵和单点积使用 Vector 专用路径。

- [赛题契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [接口与同步要点](docs/lessons.md)

本地检查：`python3 solution/tests/test_cpu.py`。它不能替代目标编译、真机 profile 和赛事成绩。

个人环境、最佳成绩证据与简要优化经验放在不提交的 `.private/`；只保留最佳赛事版本。生成物统一放 `.private/runtime/`，失败实验归纳后清理。
