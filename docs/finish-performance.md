# Finish 补偿树检查点（2026-10-03）

候选将 Cube 路径的 Finish 从逐行 Scalar Kahan 改为固定高/低分量向量补偿树。每个 batch 一次搬入 M 个 FP32 行最大值，每层 TwoSum 保留高分量加法误差，合并低分量后再次规格化。奇数节点配零，M 尾部不越界；偏移表由向量生成，UB 按 M 分配，最大自有缓冲 176 KiB + 32 字节。

Vector、Scores、分派阈值、run_kernel ABI、GM 工作内存和图捕获所有权均保持原实现。候选 kernel SHA256：`41e78bd99e00d44a35420555c05a7638b56c05fd23c6ab3cedeae854d7526768`；基线为 `70c5d65`，源码 SHA256 `64b7cd93c9cf6dce79ba9645c9dae8e269838e0240a02840c3cce94ba685270d`。当前尚待用户赛事跑分，未提 PR。

## 本次验证

- 本地默认 584/584，通过 Clang Host 地址空间、ASan/UBSan、线程集合、输入不变、输出保护区和重复确定性；最大误差/容差比 0.00373936。
- 实际 Finish 的独立 CPU 检查 252 组 / 2,100 输出，相对 math.fsum golden 全部通过；最大误差/容差比 0.0010900082。覆盖 M=1–8192 的奇偶尾部、全负、抵消、不同量级及跨 batch 的 UB 复用。
- NPU A3 环境的目标 ASC `dav-2201` 构建成功，候选含普通 C++ 动态库调用方。
- `cube-finish` 定向集合为 3 个形状 × 两类型 × 四布局，共 24 例。形状 `(1,1649,257,40)`、`(1,8191,257,40)`、`(3,8192,257,40)` 均实际进入 Cube，覆盖 M 抵消、不同量级、N 尾列和单核跨 batch 复用。
- 上述集合在 ordinary、capture-cold、capture-chain、capture-streams、shared-cold 五模式各 24/24，数值误差为 0；同时验证重复、输入和输出保护区。
- 四组带 tracing 的调用合计分配/释放各 432 个缓冲，内部 ACL 错误为 0；冷捕获和双 stream 捕获中没有 Host 内部 stream 同步。独立动态库模式不含内部 tracing。

这是针对本次改动的检查点，未执行完整设备回归或完整 stress。CPU 模型不能证明 NPU 舍入或异步同步；设备结论限于本次已测输入和调用模式。

## 同轮性能

基线与候选在同一 NPU 上顺序采集，使用相同的 24 个大型短 K 输入（SHA256 `55d9cca489f2f6ae9b1a59c3ed94a7c826a43a04fbe6573e7798d17bb1cac8ca`），两轮输出均以实际量化输入的独立 FP64 golden 核验，误差为 0。每例预热 2 次、测量 10 次。计时取 Scores 开始至 Finish 结束，包含两 kernel 间隔；每轮核对 576 条 profile 记录。

| B,M,N,K | 基线 Finish / µs | 候选 Finish / µs | 基线完整区间 / µs | 候选完整区间 / µs | 逐例加速比中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1,512,1024,128 | 14.41 | 5.07 | 118.44 | 108.69 | 1.087× |
| 1,1024,4096,128 | 27.67 | 5.76 | 426.63 | 403.25 | 1.053× |
| 1,8192,257,40 | 213.21 | 12.91 | 509.56 | 308.25 | 1.680× |

24 例均更快，最小逐例加速比约 1.041×。加速比先对每例取基线/候选，再汇总；不能直接用表中两列中位数相除。这是一轮测量，Scores 的少量变化可能包含运行波动，赛事收益需用户跑分确认。逐例数据和验证摘要见 [finish-performance-data.json](finish-performance-data.json)。

复现本地数值检查：`python3 solution/tests/test_finish.py`。设备定向数据：`python3 solution/tests/npu_data.py --suite cube-finish --prefix <prefix>`；runner 的调用模式沿用 [构建说明](../solution/README.md)。
