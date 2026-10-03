# 单次启动检查点

赛事每次迭代恰好一个 kernel。此前两次启动的 Cube 路径违反此规则。当前 Fused 在一个 MIX kernel 内完成 Scores/LongScores、AIV 汇合与 Finish；计算阶段的 TPipe 先销毁，归约阶段重新初始化以复用 UB。所有 worker 都参与一次汇合，batch 调度防止多 stream 交叠的调度死锁。

`checkpoint.py` 运行于隔离目录：`candidate/solution`、`baseline/solution` 和 `manifest.json`（两版 kernel SHA256）。基线为 `9740234`，为了对照启动规则，两版 runner 使用相同的最新测试工具。加载 CANN SDK 后运行脚本，目标 A3 / dav-2201。输入路径是既有授权实验目录，迁移环境需先恢复同一批输入。

检查实际长 K 72 例、短 K 24 例在五调用模式的精度/重复/输入和输出保护/捕获资源；另查小型 Vector 32 例。15 个自建 launch-rule 输入、每组五次迭代可复现旧版 105 次与修复版 75 次启动的差异，逐例还核对分派，不能把自建 shape 当作赛事 shape。两个固定性能集合分别含 24 例，旧版取 Scores 至 Finish 完整区间，新版取单个 Fused 完整时间；二者同轮、同输入、输出独立核验。

结果保存在 runs/results.json，失败或不完整记录不能算通过。设备结论见 [检查点报告](../../../docs/single-kernel.md)。
