# 构建与验证

比赛提交复制 [kernel.asc](kernel.asc) 全部内容，保留 `run_kernel` ABI。数值计算全部在 NPU，每次调用只启动一个 kernel。

当前实现有三个分派：

| 条件 | 计算路径 |
| --- | --- |
| M=N=1 | Vector 点积，FP32 乘法和树形 K 归约 |
| K≤128 且 M,N≤32 | Vector 小矩阵，整组输入缓存到 UB，K 并行点积 |
| 其他合法形状 | 原生完整 K 的 Cube，单个 MIX kernel 完成两级归约 |

Cube 从原始 ND 输入读取，两种转置在模板和 tiling 中同时启用。每个 worker 只保留一个得分块的 GM 暂存，不生成完整 B*M*N 矩阵。得分块按实际 N 尾宽紧密写出，再搬入 UB 时按 8 个 FP32 对齐行距。沿 batch/M/N 分工，各 N 分区写独立行最大值，所有 AIV 完成一次全核屏障后按固定顺序合并。大 M 最多分两段读取；M 求和遇到强抵消时重读行最大值并使用 TwoSum 残差累加。

宽的短/中 K 任务使用 256 列基本块。长 K 的窄矩阵使用 64 列；任务数量不足时逐步减小 M 块，最低 16 行。调度依据形状与可用核数，不依赖输入内容或赛事测试点编号。

Host 缓存只保存 tiling 与工作内存。工作内存按 context/stream 复用，容量增加时保留已有分配，避免旧图引用失效；普通调用无逐次同步释放。独立调用程序可在图销毁后、stream/context 销毁前调用 `batch_matmul_max_sum_release(stream)`。测试 runner 执行该清理并检查内部 ACL 返回值。

完整 K 的硬件 FP32 累加会在大小量级强抵消输入上丢失低位，M 求和的补偿不能修复 K 点积误差。`long-precision` 是独立精度诊断集，不能以普通随机用例通过代替该类检查。

## 本地检查

从仓库根运行，依赖 g++、clang++ 和 NumPy：

```bash
python3 solution/tests/test_cpu.py
python3 solution/tests/test_finish.py
```

CPU 模型检查索引、初始化、对齐、DMA 位置、输入不变、保护区、确定性、线程归约和每次单 launch，启用 ASan/UBSan；其 Matmul 使用 FP64 点积，不模拟硬件 Cube 舍入或异步流水。Finish 独立对照 `math.fsum`，覆盖 M≤8192、1/2/7 个 N 分区、全负、抵消及不同量级。官方 CPU Twin 本机配置与 GDB 调试见 [入口说明](tests/cpu_twin/README.md)。

## 目标设备检查

本机生成物统一写入 `.private/runtime/`。加载目标 CANN SDK 后，从仓库根运行：

```bash
cmake -S solution/tests -B .private/runtime/npu/build -DNPU_ARCH=dav-2201 \
  -DBMMMS_TRACE_RUNTIME=ON -DBMMMS_BUILD_SHARED_TEST=ON
cmake --build .private/runtime/npu/build -j4
python3 solution/tests/run_npu_checks.py --suites native
```

runner 支持普通重复、冷图捕获、同图连续调用、双 stream 图重放和普通 C++ 动态库 ABI 调用。数值通过不能代替赛事单 kernel 约束：

```bash
python3 solution/tests/npu_data.py --suite launch-rule --prefix .private/runtime/npu/data/launch-rule
cd .private/runtime/npu
msprof --output=profile --application='./build/bmmms_npu_runner data/launch-rule.bin data/profile.out.bin --profile-five'
```

`check_launch_profile.verify(csv, metadata, 5)` 按当前实际分派核验数量与名称。该自建集合的 15 组各执行 5 次，应恰好有 75 条 kernel 记录，不代表赛事未知形状。

`--benchmark` 每组预热 2 次、测量 10 次；流事件时间包含队列及提交影响，纯 kernel 时间需从同批输入的 msprof 取得。比较工具同时检查完整采样和单次启动：

```bash
python3 solution/tests/summarize_benchmark.py .private/runtime/npu/data/benchmark.json \
  .private/runtime/npu/runs/baseline.log .private/runtime/npu/runs/candidate.log \
  .private/runtime/npu/runs/comparison.json \
  --baseline-profile .private/runtime/npu/profile-baseline \
  --candidate-profile .private/runtime/npu/profile-candidate
```

比较命令从仓库根运行。`run_npu_checks.py` 的默认构建与报告目录分别为 `.private/runtime/npu/build`、`.private/runtime/npu/runs`，也可用 `--build`、`--runs` 指定绝对路径；runner 在报告目录执行，运行时附带生成物也留在该处。省略 `--suites` 运行包含强抵消输入的完整诊断，可能暴露上述硬件精度限制。

模板 smoke 入口为 `bash solution/run.sh`，构建、输入与输出位于 `.private/runtime/smoke`。日志和 profile 属于个人运行产物；保存检查结论及必要证据后，清理过期生成物。
