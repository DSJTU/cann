# 单次 kernel 启动修复（2026-10-03）

此前 Cube 路径分别启动 Scores/LongScores 与 Finish，虽然设备精度和捕获检查通过，却违反赛事 profiling 的“每次迭代恰好一个 kernel”约束。运行失败原因是启动数量被规则拒绝；不能把界面中显示的错误占比解读成数值误差。当前所有分派均只有一次 Host launch。

候选 kernel SHA256：`8edcd40966e2cd5d2b7d356b2697e12500437de0944fc90dbdbc5003db3dde69`。对照为 `9740234e7efc576471b195611ed810b261ca5053`，SHA256 `589aff63300fcfd9f153725a7de579e8cca04e58320f897838ca686913b13df5`。等待用户用修复源码重新评测，尚无修复后的赛事通过或分数结论，不提 PR。

## 修复

- Cube 只启动一个 `__mix__(1,2)` Fused。Scores/LongScores 和 Finish 改为设备函数，保持短 K 的 32×64 tile、长 K 的 64×64 tile / 128-K 补偿、N 分区及数学归约顺序。
- 每个 AIV（包括无 score 任务的 worker）在生产者 MTE3 完成后恰好执行一次 `SyncAll()`；随后读取独立行最大值，合并 N 分区并求和 M。Finish 的 batch 步长使用 `2*GetBlockNum()`，避免两个 AIV 重复写输出。
- Fused 使用 batch 调度，防止多 stream 交叠的全核汇合死锁。Scores 的 Matmul 服务和 TPipe 在设备函数返回时析构，Finish 创建新 TPipe 复用 UB；没有叠加两阶段的缓冲。Finish 最大自有 UB 仍为 176 KiB + 32 字节。
- `__kfc_workspace__` 只留在实际 kernel 入口；目标 ASC 编译明确禁止此属性用于设备函数参数。run_kernel ABI、输入、scratch 和图捕获资源所有权保持。
- CPU runner 逐次断言一个 launch；设备 profile 检查总数与逐例分派，并拒绝额外 kernel、缺失记录和错误顺序。完整回归工具也加入启动数量验收。

SDK 核查：CANN 9.0 安装实现中，TPipe 析构调用 Destroy，新构造调用 Init；Matmul 注册隔离 AIC 服务和 AIV 路径。AIV-only SyncAll 用于融合后的跨 Vector 汇合。batch 调度与 MIX 全核同步依据 [官方调度说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html)，实际正确性仍以以下目标编译和设备检查为证。

## 实际验证

- 本地默认 656/656，Host 地址空间、ASan/UBSan、布局、尾部、输入不变、保护区和重复确定性通过；逐次核验一个模拟 launch。最大误差/容差比 0.00373936。此模型不能证明真实 Cube 舍入、硬件同步或启动记录。
- 新融合路径额外检查 24 个大型短 K CPU 用例全部通过，包括大 M 和 batch 复用；独立 Finish 756 例 / 6,300 输出对照 math.fsum，通过且最大误差为 0。
- A3 / Ascend910_9362 / CANN 9.0 / dav-2201：ASC executable、共享库与独立调用方编译通过。长 K 72 例、短 K 24 例各在 ordinary、capture-cold、capture-chain、capture-streams、shared-cold 五模式全部通过；另查小型 Vector 32/32。
- 长 K 每模式最大误差/容差比 0.00116811；短 K 为 0。所有已测调用保持重复确定性、输入和输出保护。直接 runner 共分配/释放各 1,728 个缓冲，error=0；冷捕获和双 stream 捕获无 Host 内部同步。
- 15 个**自建**规则诊断输入（6 Cube、9 Vector），每组五次调用：旧版实际 **105** 个 kernel 记录，新版 **75** 个，逐例也核对单 launch 及分派。两版输出均 15/15 通过。这复现了两阶段启动造成的数量差异，不能把自建输入当成赛事隐藏 shape。
- 两个固定性能集合各 24 例，候选每例预热 2 次 / 测量 10 次，分别严格检查 **288** 个 Fused kernel 记录；无额外 Finish launch。旧版每集合 576 个 Scores/Finish 记录。双方性能输出均通过独立 FP64 golden。

## 同轮性能

以下对照为被单次启动规则拒绝的 `9740234`，旧版计 Scores 开始至 Finish 结束的完整区间，新版计单 Fused 完整时间。每形状包括两类型、四布局；加速比先逐例求，再取中位数。

| B,M,N,K | 旧版 / µs | 单次启动版 / µs | 逐例加速比中位数 |
| --- | ---: | ---: | ---: |
| 1,129,1024,8192 | 938.75 | 944.99 | 0.995× |
| 1,256,1024,1024 | 121.44 | 121.59 | 0.998× |
| 8,65,257,136 | 65.75 | 67.96 | 0.967× |
| 1,512,1024,128 | 108.88 | 107.54 | 1.013× |
| 1,1024,4096,128 | 405.19 | 401.85 | 1.006× |
| 1,8192,257,40 | 306.56 | 306.35 | 1.002× |

整体接近旧版，第三组长 K 本轮约慢 3.4%；融合增加全核汇合但满足启动约束。不能沿用旧版的赛事可提交结论，也不能据这些样本宣布修复后的赛事收益。原始报告、源码哈希、逐例性能与资源计数见 [JSON](single-kernel-data.json)，可复现检查点见 [工具说明](../solution/experiments/single_kernel/README.md)。没有重跑完整历史设备回归。
