# BatchMatmulMaxSum

Ascend C 比赛算子。提交 [solution/kernel.asc](solution/kernel.asc) 全部内容，支持 FP16/BF16、四种布局和 FP32 输出，每次调用恰好启动一个 kernel。

通用矩阵采用 AIC 直接持有 SDK Matmul，两个配对 AIV 共享双槽并归约。小矩阵使用静态 UB 和多行批量 Vector 计算；单点积单独分派。

- [赛题契约](cann_problem.md)
- [构建与验证](solution/README.md)
- [接口与同步要点](docs/lessons.md)

本地检查：`python3 solution/tests/test_cpu.py`。它不能替代目标编译、真实设备 profile 和赛事成绩。

最佳参考、成绩证据、环境及有用经验放在不提交的 `.private/`；只保留最佳赛事版本。生成物统一放 `.private/runtime/`，失败实验归纳后清除。
