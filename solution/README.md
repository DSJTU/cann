# 构建与验证

赛事提交复制 [kernel.asc](kernel.asc) 全部内容，保留 `run_kernel` ABI，每次调用恰好一个 kernel。

| 条件 | 路径 |
| --- | --- |
| M=N=1 | Vector 点积 |
| M≤32、N≤64、K≤256、N*K≤8192、M*N*K≤65536 | Vector 小矩阵 |
| 其余 | Cube 完整 K，N 窗口异步输出与 Vector 最大值归约 |

Cube 使用独立 GM 槽，窗口最大八块。只保存行最大值，不保存完整得分矩阵。跨 M/N 分区在一次 AIV 屏障后按固定顺序归约；单任务可直接输出。Host 缓存 tiling 与 context/stream scratch，容量增加时替换旧内存；不承诺旧图跨扩容保活。

## 本地检查

```bash
python3 solution/tests/test_cpu.py
```

需要 NumPy、g++、clang++。检查两种精度、四种布局、尾块、输入不变、重复性、保护区、ASan/UBSan 和模拟单 launch。异步槽在模型中同步计算，精度与流水线仍须设备验证。官方 CPU Twin 见 [说明](tests/cpu_twin/README.md)。

## 真机检查

加载 CANN SDK 后，从仓库根运行：

```bash
cmake -S solution/tests -B .private/runtime/npu/build -DNPU_ARCH=dav-2201 -DBMMMS_TRACE_RUNTIME=ON
cmake --build .private/runtime/npu/build -j4
python3 solution/tests/run_npu_checks.py
```

默认做相关边界数值、普通调用、固定形状冷捕获和真实单 kernel profile；其他模式按改动选择。`--benchmark` runner 预热两次并测十次，纯 kernel 耗时以 msprof 为准。

生成物放 `.private/runtime/`。自建形状仅用于筛选优化，不能推断未知赛事形状或声称赛事分数已提升。
