# 构建与验证

赛事提交复制 [kernel.asc](kernel.asc) 全部内容。保持 `run_kernel` ABI、输入不变和重复确定性，每次调用恰好启动一个 kernel。

| 条件 | 路径 |
| --- | --- |
| M=N=1，K≤64 | 静态 UB 点积，K=32/64 编译期特化 |
| M=N=1，K>64 | Vector 点积，队列管理 UB |
| M,N≤64、K≤1024，满足输入、乘积及 180KiB UB 容量限制 | 静态 UB 小矩阵，多行批量发射与树形归约 |
| 其余 | MIX(1,2)，AIC 直接使用 SDK Matmul、双 AIV 归约 |

精确分派条件以源码为准。通用路径在 M/N 维划分任务，完整 K 点积后取 N 最大值，再求和 M。每个 Cube 使用两个共享 GM 槽，两个 AIV 各处理一半行；只有单行任务、无 N 分区时由 AIV0 直接输出。其他情况在全 AIV 屏障后归约分区结果。

Host 缓存 tiling 和 context/stream scratch，按实际需求分配，不预留大块空间或保存旧图。转置属性沿用调用输入。计算结果和输入不缓存。

## 本地检查

```bash
python3 solution/tests/test_cpu.py
```

需要 NumPy、g++、clang++。检查两种精度、四种布局、边界、输入不变、重复性、输出保护区、ASan/UBSan 和模拟单 launch。数学模型以同步 FP64 点积检查地址及控制流，配对线程模型检查共享槽握手；不模拟 NPU 流水、硬件舍入、图捕获或性能。

测试工具本身的错误证据回归：`python3 solution/tests/test_validation.py`。新增语义边界可单跑 `python3 solution/tests/test_cpu.py --suite robustness`。

## 真机检查

加载 CANN 9.0 SDK 后，从仓库根运行：

```bash
cmake -S solution/tests -B .private/runtime/npu/build -DNPU_ARCH=dav-2201 -DBMMMS_TRACE_RUNTIME=ON
cmake --build .private/runtime/npu/build -j4
python3 solution/tests/run_npu_checks.py --suites baseline --modes ordinary
```

上述最小检查点包含四类入口、存储布局、重复调用及真实单 kernel profile。扩大边界覆盖可用 `--suites native`；固定形状图捕获可用 `--modes capture-cold`，其他调用模式按改动选择。图内 scratch 扩容不作为比赛优化目标，需单独验证。

`--benchmark` runner 预热两次并测十次；赛事相关性能比较使用同设备同输入的 msprof kernel duration。生成物放 `.private/runtime/`，记录被验证源码哈希，自建计时不能代替赛事分数。详见[性能与用例方法](tests/README.md)。
