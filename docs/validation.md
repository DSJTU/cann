# 2026-10-02 验证记录

本页记录 Git `d67572d` 的精度修复与调度优化版本，kernel SHA-256 为 `a594fb00ac0ffd26dee750310dbb7944d23687876f9130cb297ffb0807815edb`。后续内核的性能优化与验证见 [性能记录](performance.md)；下文“当前”“最终”均指本页记录的版本。

当前版本修复跨 K 块累加丢失小数项的问题，并减少部分大任务的同步参与核。CPU 模型、目标 ASC 编译、官方仿真和 A3 真机验证均已执行。用户于 2026-10-02 提供当前版本的赛事结果截图：15/15 Pass，各点输出错误占比均为 0.00%。

机器可读的真机回归摘要及 48 组逐项计时见 [validation-data.json](validation-data.json)。本记录中的性能结果来自同一台设备上的独立测量，不能作为赛事分数。

## 版本与环境

| 版本 | kernel.asc SHA-256 | 用途 |
| --- | --- | --- |
| 原基线，Git `6199aaf453ef7922a5715786730401fed7f13fcf` | `410f6f459c5880f4381ef8c2e57950bfd3a0662cda5d9b315ddfbb9e02144b85` | 故障复现与性能对照 |
| 仅精度修复 | `4b0c1fb2f1b2fb6154eea265d5eca5cbc25d0c8670de7fef5e74efa13a2f08af` | 仿真验证与拆分性能代价 |
| 精度修复及调度优化，当前版本 | `a594fb00ac0ffd26dee750310dbb7944d23687876f9130cb297ffb0807815edb` | 最终 CPU、目标编译及真机回归 |

目标环境为 ARM64、CANN 9.0，ASC 构建架构 `dav-2201`；真机为 A3，ACL 报告 SoC `Ascend910_9362`、可用 Cube 核数 20。官方 camodel 使用 `Ascend910B3`。基线、精度版本与最终版本分别在独立目录构建，原有工作目录未被覆盖。

## 故障与修复

对形状 `(B,M,N,K)=(1,9,17,8192)`，令 A 全为 1；B 每列沿 K 的四段分别为 2048 个 1、2048 个 `2^-20`、2048 个 -1、2048 个 0。量化到 FP16 或 BF16 后这些值仍精确可表示，正确点积为 `2^-9`，最终输出为 `0.017578125`。原基线真机输出为 0，原因是跨 128 元素 K 块的 FP32 直接累加先丢失小数项，随后大数抵消。

新增 96 组集合覆盖长 K 抵消、大小数混合、相近列最大值、尾块与 7/19/20 核分区，包含两种输入类型及四种布局。原基线真机通过 64/96，失败 32 组；最坏误差/容差为 9473.6842。修复后的精度版本与最终版本均通过 96/96。

内核对 `K>128` 使用向量 Kahan 补偿累加 K 块结果，新增 512 字节 UB 缓冲，并复用已完成局部归约的 `highParts`。保留单块 K 的直接累加路径，补齐向量依赖屏障；`run_kernel` ABI、输入和 GM 工作内存约定保持不变。该修复针对跨块累加，块内乘法和归约仍存在 FP32 舍入。

大任务分区原先取 `groups=min(cores/batch,M)`。现在先计算 `rowsPerGroup=ceil(M/groups)`，再取 `groups=ceil(M/rowsPerGroup)`，保持每核最大行数不增加，减少同步参与核。例如 20 核下 M=64 从 20 个分区降为 16 个，M=129 降为 19 个。固定顺序归约、硬件同步和一次 kernel 启动保持不变。

## 已执行验证

误差指标为 `abs(actual-golden)/(1e-4+1e-4*abs(golden))`，不超过 1 即通过。Golden 从实际量化输入以 FP64 独立计算，最后转 FP32。

| 检查 | 版本 / 集合 | 结果 |
| --- | --- | --- |
| 本地 CPU 模型 | 最终版本，默认 376 组 | 376/376；ASan、UBSan、Clang 地址空间检查通过；最坏误差/容差 0.00373936 |
| 目标 ASC 编译 | 最终 runner、共享库、公开 main 模板 | 构建成功，运行验证完成，最终脚本状态 0 |
| 官方 camodel | 仅精度修复版本，smoke 32 组 | 32/32；最坏误差/容差 0.000698916 |
| 官方 camodel | 仅精度修复版本，紧凑长 K 抵消 8 组 | 8/8，误差为 0 |
| A3 普通调用 | 最终版本，原有 correctness 280 组 | 280/280；最坏误差/容差 0.00373936 |
| A3 普通调用 | 最终版本，extended 96 组 | 96/96；最坏误差/容差 0.00104019 |
| A3 冷捕获、同图连续三次、双 stream | 最终版本，每种 extended 96 组 | 每种 96/96 |
| 普通 C++ 调用独立共享库 | 最终版本，普通、冷捕获、双 stream，每种 extended 96 组 | 每种 96/96 |
| 性能运行与 msprof 采集 | 原基线、仅精度修复、最终版本，每轮 benchmark 48 组 | 各轮输出验证通过，采集成功 |

CPU 与真机 runner 同时检查输入不变、输出保护区及重复确定性；双 stream 检查两个独立输出。官方仿真验证的是仅精度修复版本，最终版本的 host 分区变化通过目标编译及上述真机调用模式验证。

这些集合覆盖所列边界和调用模式，未穷举整个合法尺寸空间；大型 `stress` 集合不计入本次完成项。msprof 的流水线利用率字段为 N/A，因此本次没有据此给出算子瓶颈结论。

## 性能结果

benchmark 固定随机种子 20261002，6 个代表形状 × 2 种输入类型 × 4 种布局，共 48 组，使用设备可用的 20 核。各版本输入文件 SHA-256 相同：`b0e350b028f484d0efb9355f9d2ca77168fc013129a933a56451e2bb8b60f684`。每组预热 2 次，随后测量 10 次取中位数；流事件与 msprof 分别运行。以下均为 msprof `Task Duration(us)` 的逐用例比值中位数，小于 1 表示更快。

| 形状 `(B,M,N,K)` | 精度修复 / 原基线 | 最终 / 仅精度修复 | 最终 / 原基线 |
| --- | ---: | ---: | ---: |
| (1,64,257,128) | 1.0006 | 0.9499 | 0.9505 |
| (1,129,257,136) | 1.0566 | 0.9884 | 1.0401 |
| (1,9,17,8192) | 1.0252 | 0.9973 | 1.0161 |
| (3,21,129,392) | 1.0626 | 1.0073 | 1.0674 |
| (8,33,65,128) | 1.0071 | 0.9821 | 0.9892 |
| (64,3,17,32) | 1.0091 | 1.0488 | 1.0501 |
| 全部 48 组 | 1.0157 | 0.9961 | 1.0203 |

精度修复整体约增加 1.6% kernel 时间；调度变化在 `(1,64,257,128)` 上约减少 5%，最终版本相对原基线的整体中位比值约增加 2.0%。全部行是对逐项比值重新取中位数，不是对形状中位数再平均。未改变分区的形状也存在运行间波动；例如最后一行形状的变化不能归因于调度优化。这是一轮测量，没有置信区间。

ACL 流事件的整体比值为精度修复/原基线 1.0164、最终/精度修复 0.9971；事件区间包含队列和提交影响，逐项数据见 JSON，应与设备 kernel 时间区分。

## 复现入口

本地完整检查：

```bash
python3 solution/tests/test_cpu.py
```

设备构建、输入生成和输出验证见 [solution/README.md](../solution/README.md)。`npu_data.py` 的 `correctness`、`extended`、`precision`、`benchmark` 分别生成上述集合；runner 支持 `--capture-cold`、`--capture-chain`、`--streams`、`--benchmark`，共享库由 `BMMMS_BUILD_SHARED_TEST=ON` 构建。

性能采集示例（在 solution 目录，SDK 环境已加载）：

```bash
python3 tests/npu_data.py --suite benchmark --prefix lab-data/benchmark
mkdir -p runs
./build-npu/bmmms_npu_runner lab-data/benchmark.bin lab-data/benchmark.out.bin \
  --benchmark 2> runs/benchmark.log
python3 tests/npu_data.py --prefix lab-data/benchmark --verify lab-data/benchmark.out.bin
msprof --output=lab-profile --aic-metrics=PipeUtilization \
  --application='./build-npu/bmmms_npu_runner lab-data/benchmark.bin lab-data/profile.out.bin --benchmark'
python3 tests/npu_data.py --prefix lab-data/benchmark --verify lab-data/profile.out.bin
```

每个版本使用独立构建和采集目录；用 `tests/summarize_benchmark.py` 比较两轮事件日志及 profile。工具检查每例恰有 10 个不重复的计时样本，并检查 profile 的 12 次 kernel 调用数。
