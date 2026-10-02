# BatchMatmulMaxSum

正确性基线已通过赛事 15/15，见 [验证记录](../docs/validation.md)。当前性能优化内核的真机验证与计时见 [性能记录](../docs/performance.md)，尚未取得该版本的赛事结果。提交复制 [kernel.asc](kernel.asc) 全部内容，保留模板 run_kernel ABI；数值计算全部在 NPU。

小任务按 batch 分核，大任务沿 M 分区；保持最大行分区长度不增加，减少不必要的同步参与核。各核完成 FP32 点积及跨 K 块补偿累加、N 最大值、分区补偿求和，再通过硬件屏障与输出位置按固定顺序汇总。一次 kernel 启动，无额外 GM 工作内存或原子累加。

N 分块按布局取 16/32/64 列；超过 8 个有效列时使用向量最大值归约。K≤128 且 M、N≥16 时，各核复用 B 分块并一次搬入最多 16 行 A，将行最大值暂存在 UB，最后仍按原行顺序补偿求和。ND-B 尾块按 16 元素对齐实际搬运 pitch，单侧填充不超过 32 字节。

在仓库根运行本地检查，依赖 g++、clang++、NumPy：

```bash
python3 solution/tests/test_cpu.py
```

默认 440 组：原有 280 组，加 96 组长 K 精度与分区边界、64 组列分块与批量 A 行边界；覆盖两种类型、四种布局、输入不变、输出保护区与确定性，启用 ASan/UBSan。用 `--suite correctness`、`--suite extended` 或 `--suite performance` 可分开运行。类型模型与 CPU 模型不能代替目标 SDK 编译和真机验证。

NPU 验证工具位于 [tests/](tests/npu_runner.asc)：独立 FP64 golden、普通/冷捕获/同图连续调用/双 stream、普通 C++ 动态库调用。构建前加载目标 CANN SDK 环境。

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
