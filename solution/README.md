# 构建与验证

赛事提交复制 [kernel.asc](kernel.asc) 全部内容，保留 `run_kernel` ABI，每次调用恰好一个 kernel。

| 条件 | 路径 |
| --- | --- |
| M=N=1 | Vector 点积 |
| 很小的矩阵，或高 batch 的既有小矩阵范围 | Vector 小矩阵 |
| K≥4096、M≤128、N≤256、B≤8，行面板能固定到核且重排有益 | 核内 NZ 重排、完整 A 面板 L1 复用、直接 Cube 指令 |
| K≤256，行任务能固定到核，且布局/网格满足分派条件 | 原生 ND 搬入 L1，直接 Cube 指令 |
| 其余 | SDK Matmul，N 窗口异步输出 |

M=1 的 A、N=1 的 B 两种存储布局分别等价，Host 选择连续输入布局。直接路径使用 MIX(1,1)，跳过 Matmul 服务与请求协议；SDK 路径使用 MIX(1,2)，窗口最多八块，两条路径使用独立调度参数。直接路径每核固定一个行任务，长 K 重排和计算都在同一个 kernel 内完成。重排保留 FP16/BF16 原始位，每次调用重新读取输入；K 尾部补零，使每次 Mmad 使用相同的 K 面板长度。

Cube 完成 K 点积后，Vector 才归约 N 最大值。输出使用独立 GM 槽握手，只保存行最大值与必要分区结果，不保存完整得分矩阵。最终 M/N 分区按固定顺序归约。Host 缓存 tiling 与 context/stream scratch，一次预留 64MiB；容量增加时替换旧内存，不保留旧图或输入结果缓存。核数限制在设备物理并发能力内。

## 本地检查

```bash
python3 solution/tests/test_cpu.py
```

需要 NumPy、g++、clang++。检查两种精度、四种布局、尾块、输入不变、重复性、保护区、ASan/UBSan 和模拟单 launch。数学模型用同步 FP64 点积，不模拟硬件流水、舍入与耗时；不能替代目标编译和真机检查。官方 CPU Twin 见 [说明](tests/cpu_twin/README.md)。

## 真机检查

加载 CANN SDK 后，从仓库根运行：

```bash
cmake -S solution/tests -B .private/runtime/npu/build -DNPU_ARCH=dav-2201 -DBMMMS_TRACE_RUNTIME=ON
cmake --build .private/runtime/npu/build -j4
python3 solution/tests/run_npu_checks.py
```

默认检查相关边界、普通调用、固定形状冷捕获和真实单 kernel profile；其他调用模式按修改选择。`--benchmark` runner 预热两次并测十次，纯 kernel 耗时以 msprof 为准。profile 检查使用 runner 记录的实际核数识别分派。

生成物放 `.private/runtime/`。自建形状仅用于筛选优化，不能推断未知赛事形状或声称赛事分数已提升。
