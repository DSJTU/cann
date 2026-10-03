# BatchMatmulMaxSum

此前 Vector 正确性基线和性能优化版本均通过赛事 15/15，见 [验证记录](../docs/validation.md) 和 [Vector 性能记录](../docs/performance.md)。当前混合实现的验证独立记录在 [Cube 性能记录](../docs/cube-performance.md)。提交复制 [kernel.asc](kernel.asc) 全部内容，保留模板 run_kernel ABI；数值计算全部在 NPU。

K≤128、M/N≥256、M*N*K≥2^24 时使用 Cube 的 32×64 得分块并立即沿 N 归约；第二个 Vector kernel 按 M 原顺序补偿求和。除系统工作内存外只分配 B*M 个 FP32 行最大值；普通调用同步释放，图捕获在模型销毁时释放，同图多次调用使用独立缓冲。

其他尺寸保留一次启动的 Vector 路径：小任务按 batch 分核，大任务沿 M 分区。各核完成 FP32 点积、跨 K 块补偿、N 最大值和分区求和，再按固定顺序跨核汇总，无额外 GM 工作内存。

N 分块按布局取 16/32/64 列。K≤128 且 M、N≥16 时，B 分块搬入一次后供本核的多行使用；分区行数不超过 128 且 N 跨多个分块时，A 也只转换一次。点积按行紧密排列，向量归约得到每行最大值，标量补偿求和仍按原行序进行。K>128 且分区多于一行时，最多 16 行共用每个 B 的 K/N 分块，跨 K 仍按元素做补偿累加，全部 K 完成后再取 N 最大值。单行长 K 保持逐行路径。ND-B 尾块按 16 元素对齐实际搬运 pitch，单侧填充不超过 32 字节。

在仓库根运行本地检查，依赖 g++、clang++、NumPy：

```bash
python3 solution/tests/test_cpu.py
```

默认 584 组：280 组正确性，184 组长 K 与分区（含多行复用 B 的抵消和尾列），120 组分块边界（含 A 缓存 128/129 行边界，以及 16 组 Cube 分派与 worker 复用）；覆盖两种类型、四种布局、输入不变、输出保护区与确定性，启用 ASan/UBSan。用 `--suite correctness`、`--suite extended` 或 `--suite performance` 可分开运行。CPU Matmul 使用 FP64 点积，只验证分区、索引、打包与归约控制流，不模拟 Cube 硬件舍入、流水线或图捕获。类型模型与 CPU 模型不能代替目标 SDK 编译和真机验证。

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

旧 `summarize_benchmark.py` 仅支持单个 Vector kernel。Cube 的性能比较使用 `compare_cube_profile.py`，将 Scores 开始到 Finish 结束作为整个算子的设备区间，包含 kernel 间隔；输入与数值验证必须对应同一批数据。
