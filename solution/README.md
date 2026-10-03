# BatchMatmulMaxSum

提交复制 [kernel.asc](kernel.asc) 全部内容，保留模板 run_kernel ABI；数值计算全部在 NPU。本分支仍在开发。当前扩大短 K Cube 覆盖的源码尚未赛事评测或设备回归，不能继承此前面板批量版本的赛事通过或 `main` 上 `54cc8bd` 的确认提升。逐点耗时只记在本地 `.private`。接口约束见 [Ascend C 开发要点](../docs/lessons.md)，更早的 Vector 与短 K 记录见 [文档索引](../docs/README.md)。

K≤128、M≥16、N≥64、B*M*N*K≥2^20 时使用短 K Cube。基本块仍是 32×64；一次请求最多覆盖 1024 列，Iterate 按 N 交回每个基本块并立刻归约。batch/M 任务不足时再沿 N 分区，让更多核计算独立区间。同一个 MIX kernel 的 AIV 在全核汇合后先逐行合并 N 分区最大值，再以固定高/低分量补偿树求和 M。每层用 TwoSum 保留加法舍入误差，再合并低分量并规格化；奇数节点补零。自有 UB 按 M 分配，最大 176 KiB + 32 字节。除系统工作内存外只分配 B*M*nGroups 个 FP32 行最大值；普通调用同步释放，图捕获在模型销毁时释放，同图多次调用使用独立缓冲。

K>128、M≥16、N≥64、B*M*N*K≥2^22 时使用长 K Cube。每个任务把一个 64×64 块的全部 K 面板打包成连续 ND 矩阵，一次 IterateBatch 算出所有 128 长的面板，再逐元素补偿，全部面板结束后才取该块的 N 最大值。Cube 累加长度仍是 128；完整 K 和 512-K 累加在设备抵消用例上失败。任务按 M 与受限的 N 分区分配，每个分区写独立行最大值，同一个 kernel 中的 Finish 先合并 N 分区，再以补偿树求和 M。自有 GM 为 B*M*nGroups 个 FP32 行最大值，另有按核复用的面板打包缓冲；N 分区由核数限制。

其他尺寸保留一次启动的 Vector 路径：小任务按 batch 分核，大任务沿 M 分区。各核完成 FP32 点积、跨 K 块补偿、N 最大值和分区求和，再按固定顺序跨核汇总，无额外 GM 工作内存。

N 分块按布局取 16/32/64 列。K≤128 且 M、N≥16 时，B 分块搬入一次后供本核的多行使用；分区行数不超过 128 且 N 跨多个分块时，A 也只转换一次。点积按行紧密排列，向量归约得到每行最大值，标量补偿求和仍按原行序进行。K>128 且分区多于一行时，最多 16 行共用每个 B 的 K/N 分块，跨 K 仍按元素做补偿累加，全部 K 完成后再取 N 最大值。单行长 K 保持逐行路径。ND-B 尾块按 16 元素对齐实际搬运 pitch，单侧填充不超过 32 字节。

在仓库根运行本地检查，依赖 g++、clang++、NumPy：

```bash
python3 solution/tests/test_cpu.py
```

默认 896 组：280 组正确性、184 组长 K 与分区、136 组分块边界、192 组短 K Cube 分派与 N 分区边界，以及 104 组长 K Cube 分派边界用例。短 K 覆盖 1024 列窗口、分区余数、不同 N 分区获胜、M/N 尾部、batch 总工作量门槛，以及门槛以下的 Vector 回退；新增集合在两类型、四布局、1/20 核下分别运行。长 K 覆盖一次批量请求内的多个 128 面板、512-K 分段内部抵消，以及 M=16、N=64 的分派边界。覆盖输入不变、输出保护区与确定性，逐次断言恰好一个 launch，启用 ASan/UBSan。用 `--suite correctness`、`--suite extended`、`--suite performance`、`--suite short-cube` 或 `--suite long-cube` 可分开运行。CPU Matmul 使用 FP64 点积，只验证分区、索引、打包与归约控制流，不模拟 Cube 硬件舍入、流水线或图捕获。类型模型与 CPU 模型不能代替目标 SDK 编译和真机验证。

Finish 的独立数值检查为 `python3 solution/tests/test_finish.py`（仓库根运行），756 组 / 6,300 个输出对照 `math.fsum`，覆盖 1/2/7 个 N 分区、M 尾部、多 batch 复用、全负、抵消和不同量级。`--suite cube-finish` 可用于 CPU runner 或 NPU 数据生成器，提供 24 个实际进入短 K Cube 的大 M 定向用例；它单独运行，不计入默认 896 组。

赛题日常迭代做相关本地检查并正常提交；关键检查点先做最小必要设备回归，再由用户跑分。确认提升后才提 PR。下方完整回归工具按需使用。

NPU 验证工具位于 [tests/](tests/npu_runner.asc)：独立 FP64 golden、普通/冷捕获/同图连续调用/双 stream 图重放、普通 C++ 动态库调用。构建前加载目标 CANN SDK 环境。

```bash
cd solution
cmake -S tests -B build-npu -DNPU_ARCH=dav-2201 \
  -DBMMMS_TRACE_RUNTIME=ON -DBMMMS_BUILD_SHARED_TEST=ON
cmake --build build-npu -j4
python3 tests/npu_data.py --suite correctness --prefix lab-data/correctness
./build-npu/bmmms_npu_runner lab-data/correctness.bin lab-data/result.out.bin
python3 tests/npu_data.py --prefix lab-data/correctness --verify lab-data/result.out.bin
```

新增精度集合使用 `--suite extended`，分块边界使用 `--suite performance`；官方仿真可用 `--suite precision` 生成 8 组紧凑的长 K 抵消用例。`--suite benchmark` 生成 48 组代表尺寸，`--suite large-short-k` 生成 24 组大型短 K 用例（完整 `stress` 集合的前三个形状）；runner 的 `--benchmark` 模式每组预热 2 次、测量 10 次，同时检查数值输出、确定性和保护区。流事件区间含队列及提交影响，设备 kernel 时间需另用 msprof 获取。

相同数据的两次计时日志可用以下工具比较，工具拒绝缺失或重复的采样：

```bash
python3 tests/summarize_benchmark.py lab-data/benchmark.json \
  runs/baseline-benchmark.log runs/candidate-benchmark.log runs/comparison.json \
  --baseline-profile lab-profile-baseline --candidate-profile lab-profile-candidate
```

profile 参数可省略；提供时分别指向仅含本轮一次 msprof 采集的目录，采集对象为 benchmark runner，每组 12 次调用。

搬运、同步与精度注意事项见 [Ascend C 开发要点](../docs/lessons.md)。

完整目标回归可用下列脚本（需开启 tracing 与 shared target），逐组核验数值、重复、保护区、内部 ACL 返回值与缓冲生命周期：

```bash
python3 tests/npu_data.py --suite large-short-k --prefix lab-data/large-short-k
python3 tests/run_npu_checks.py --build build-npu --runs runs/hybrid \
  --large-prefix lab-data/large-short-k
```

`summarize_benchmark.py` 支持当前 Baseline/Fused 单次启动的计时比较。`compare_cube_profile.py` 保留给历史两次启动实验。`check_launch_profile.py` 按 kernel 里现在的门槛区分 Fused 与 Baseline：短 K 是 M≥16、N≥64 且 B*M*N*K≥2^20，每次调用分配工作内存和行最大值；长 K 是 M≥16、N≥64 且 B*M*N*K≥2^22，另加一块按核复用的面板缓冲。`--suite launch-rule` 生成 15 个自建输入，runner `--profile-five` 每组执行 5 次，应恰好得到 75 个 kernel 记录。该集合在这套门槛下仍是 6 个 Cube、9 个 Vector，不能视为赛事未知 shape。当前性能取单个 Fused/Vector kernel 的完整区间；输入与数值验证必须对应同一批数据。
