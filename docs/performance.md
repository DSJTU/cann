# 2026-10-02 性能优化

已将通过真机验证的 Vector 优化整合到 `solution/kernel.asc`。相对正确性基线 `d67572d`，48 组代表用例的逐例 kernel 耗时比值中位数为 0.6637，24 组大型短 K 用例为 0.2607；本轮各组均更快。当前优化内核尚未取得赛事结果，已有赛事 15/15 截图对应基线。逐项耗时、版本哈希与全部数值报告见 [performance-data.json](performance-data.json)。

## 实现

保持一次 kernel 启动、原 `run_kernel` ABI、纯 NPU 计算和无额外 GM 工作内存。点积归约顺序、跨 K 块补偿累加、按行补偿求和与固定顺序跨核汇总保持一致。

1. 扩大 N 分块：ND-B 为 32 列，转置存储 B 为 64 列；N≤16 时使用 16 列。转置 B 无需分配完整转置缓冲及偏移表。
2. 向量求最大值：超过 8 个有效列时 Gather 紧凑点积结果，使用 WholeReduceMax，只读回一个 FP32 标量。短尾块仍只包含有效列。
3. 短 K 复用 B：K≤128 且 M、N≥16 时改为 N 分块在外，将 B 分块搬入、转换一次，再用于各核负责的多行 A。每行最大值保存在 UB，最后按原行顺序求和。
4. 批量 A：该路径每次搬入最多 16 行 A，兼容两个 A 存储布局；只初始化实际使用的偏移表，减少小尺寸的设置成本。

ND-B 短尾块的物理 pitch 为 `ceil(columns/16)*16`。当其小于逻辑分块宽度时使用相应 stride 的 Gather，Cast 仅访问真实搬入并填充的数据。初次扩大分块遗漏了 DataCopyPad 单侧填充的 32 字节上限，造成真机 507035；已修正并将限制加入 CPU 模型。接口限制见 [官方 DataCopyPad 文档](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html)。按源码逐缓冲计算，合法最大尺寸下 UB 不超过 140448 字节，低于模型使用的 192 KiB 预算。

## 测量方法与阶段比较

ARM64、CANN 9.0、ASC `dav-2201`，A3 SoC `Ascend910_9362`、20 个可用 Cube 核。基线与每个候选分别构建，使用同一份实际量化输入和独立 FP64 golden；每例预热 2 次，再测 10 次取中位数。ACL 流事件与 msprof 分别运行并核验输出；下表使用 msprof `Task Duration(us)`。

48 组 benchmark 为 6 个形状 × 2 种类型 × 4 种布局；输入 SHA-256 为 `b0e350b028f484d0efb9355f9d2ca77168fc013129a933a56451e2bb8b60f684`。

| 候选 | 新增变化 | kernel / 基线中位比值 | 流事件 / 基线中位比值 |
| --- | --- | ---: | ---: |
| wide | 扩大列分块，合法短尾块搬运 | 0.8309 | 0.8456 |
| wide-max | 再加向量最大值 | 0.7741 | 0.7910 |
| cache | 再加短 K 的 B 复用 | 0.7562 | 0.7794 |
| cache-block，当前 | 再加批量 A、偏移初始化缩减与小 N 分块 | 0.6637 | 0.6882 |

当前 kernel SHA-256：`bb89ad3cba93c280286be5c32b6f9a1fd2b0aa665d46d22ba2db983e38cb669e`。基线 SHA-256：`a594fb00ac0ffd26dee750310dbb7944d23687876f9130cb297ffb0807815edb`。所有中间版本的哈希见 JSON；只有最后版本整合到当前代码。

| 形状 `(B,M,N,K)` | 基线耗时中位数 µs | 当前耗时中位数 µs | 逐例耗时比值中位数 |
| --- | ---: | ---: | ---: |
| (1,64,257,128) | 112.728 | 63.009 | 0.5537 |
| (1,129,257,136) | 296.154 | 191.721 | 0.6557 |
| (1,9,17,8192) | 332.658 | 258.970 | 0.7682 |
| (3,21,129,392) | 169.017 | 134.728 | 0.7958 |
| (8,33,65,128) | 243.260 | 70.129 | 0.3070 |
| (64,3,17,32) | 47.284 | 35.445 | 0.7347 |

大型短 K 集合是完整 stress 集合的前三个形状，共 24 组，同样覆盖两种类型及四种布局。输入 SHA-256 为 `55d9cca489f2f6ae9b1a59c3ed94a7c826a43a04fbe6573e7798d17bb1cac8ca`。

| 形状 `(B,M,N,K)` | 基线耗时中位数 µs | 当前耗时中位数 µs | 逐例耗时比值中位数 |
| --- | ---: | ---: | ---: |
| (1,512,1024,128) | 2076.484 | 565.959 | 0.2850 |
| (1,1024,4096,128) | 14059.834 | 4157.707 | 0.3015 |
| (1,8192,257,40) | 7133.658 | 1342.878 | 0.1886 |

比值先对每例计算，再取中位数，不能直接用表中的两个耗时中位数相除。48 组最差比值为 0.9453，24 组最差比值为 0.3320。阶段之间的小幅变化可能包含运行波动；这是一轮测量，没有置信区间。ACL 事件包含队列与提交影响，不应代替纯 kernel 时间，也不能直接推算赛事分数。流水线利用率仍为 N/A，本记录的瓶颈判断来自代码的数据搬运与计算结构。

## 已执行验证

| 检查 | 范围 | 结果 |
| --- | --- | --- |
| CPU 模型、Clang 地址空间、ASan/UBSan | correctness 280 + extended 96 + performance 64 | 440/440 |
| 目标 ASC 构建 | 独立 runner、共享库与 C++ 调用方 | 构建成功 |
| 真机普通调用 | 同上三组集合，合计 440 | 440/440 |
| benchmark 与独立 msprof 采集 | 每轮 48 组 | 各 48/48 |
| 冷捕获、同图连续三次、双 stream | 每种 extended 96 组 | 各 96/96 |
| 普通 C++ 加载共享库：普通、冷捕获、双 stream | 每种 extended 96 组 | 各 96/96 |
| 大型短 K benchmark、独立采集、冷捕获、双 stream | 每轮 24 组 | 各 24/24 |

当前版本所有已测数值报告的最坏误差/容差不超过 0.00373936。runner 同时核验输入不变、输出保护区及重复确定性。未对当前版本重跑官方 camodel，未执行完整 48 组 stress，也未穷举尺寸空间。

复现使用 [构建与验证说明](../solution/README.md)。分别用 `npu_data.py --suite benchmark` 和 `--suite large-short-k` 生成数据，在相同设备分别运行基线与当前代码的 `--benchmark` 和 msprof，再用 `summarize_benchmark.py` 比较。每轮使用独立采集目录，输出都需通过 `npu_data.py --verify`。

## 下一步：Cube 分块与融合归约

大矩阵仍需逐行逐列执行 Vector 点积。当前优化减少搬运、转换及标量访问，计算次数仍随 M×N×K 增长；大型用例仍处于毫秒级。下一阶段优先原型化短 K、较大 M/N 的 Cube 路径，再决定长 K 的精度与分块策略。

计划按以下顺序验证：

1. 核对目标 CANN 9.0 SDK 中 Matmul tiling、FP16/BF16 与四布局接口，先完成单分块真机原型。
2. 以有界 M/N 分块计算完整 K 点积，在 N 分块之间更新行最大值，完成后再对 M 求和；不将全部 B×M×N 中间分数写入 GM。高阶 Matmul 支持将切片交给 VECIN 继续向量处理，见 [GetTensorC](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html) 和 [官方混合算子示例](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0050.html)。
3. 保持公开 ABI，明确系统 workspace 和用户 workspace 的大小及生命期。核直调需由 Host 管理 workspace，见 [官方开发仓库说明](https://asc.gitcode.com/guide/programming_guide/appendix/common_operations/how_to_use_workspace.html)；目标 SDK 能力仍需实际确认。图捕获中禁止过早释放，多 stream 需要独立使用中的缓冲，不能共用正在执行的临时内存。
4. 先复用本次全部数值与调用模式验证；针对 Cube 的累计顺序另加抵消和相近最大值测试，再比较纯 kernel 与完整调用成本。长 K 如需分块补偿，应在完成所有 K 块后取 N 最大值。
5. 按实测确定 Vector/Cube 的尺寸分界，再走赛事验证。此方案尚未实现，不计入上述性能收益。
