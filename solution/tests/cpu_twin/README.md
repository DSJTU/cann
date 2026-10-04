# 官方 CPU Twin

使用 CANN 9.0 的 `tikicpulib` 在 CPU 执行同一 Ascend C 计算体，无需 NPU 或驱动。生成器直接读取 [kernel.asc](../../kernel.asc)，适配 Host launch 和共享 GM 分配；未知 launch 结构会拒绝生成，生成清单记录源码 SHA256。

需要对应 CPU 架构的 CANN Toolkit、C++17 编译器、CMake，以及有 NumPy 的 Python；调试另需 GDB。配置 SDK 根（含 `set_env.sh`），Python 和输出目录可按需指定：

```bash
export BMMMS_CANN_ROOT=/绝对路径/cann
export BMMMS_TWIN_PYTHON=/绝对路径/python
export BMMMS_TWIN_WORK=/绝对路径/twin-work
bash solution/tests/cpu_twin/run.sh --suite all
bash solution/tests/cpu_twin/run.sh --suite long-precision
bash solution/tests/cpu_twin/run.sh --suite long-cube --case 8 --gdb
```

Python 默认取 PATH 中的 `python3`，SDK 也可由 `ASCEND_HOME_PATH` 提供；`CXX` 可指定编译器。工作目录默认为仓库 `.private/runtime/cpu-twin`，包含 build、data 和运行日志。

验证成功后，执行日志压缩为 `data/<集合>.log.gz`，清除 SDK 自动生成的 `npuchk/`、`cceprint/` 和 `stub_reg.log`。失败或 GDB 调试时保留诊断现场；确认原因后再清理，避免日志在项目根目录散落。

| 集合 | 覆盖 |
| --- | --- |
| smoke | 点积与小矩阵 |
| small-matrix | 小矩阵边界、多 batch 复用 |
| short-cube / long-cube | 完整 K、M/N 尾块 |
| all | 上述普通集合，包含两种类型与四种布局 |
| long-precision | 大小量级抵消诊断，独立于 all |

每组重复两次，检查输入不变、保护区、确定性、一个 CPU launch，并对照实际存储值的 FP64 golden。MIX 模式使用 1 个 AIC 与 2 个 AIV 进程；精度诊断另覆盖 1/2 个逻辑核。完整 K 的 FP32 累加有强抵消限制，普通集合通过不能代替该项诊断。

GDB 中使用 `break kernel.asc:行号` 或在 `FusedMaxSim::Process`、`BmmmsSmallKernel::Process` 对应行设断点。入口跟随 fork 子进程，保留其他进程并启用 `schedule-multiple`，让核间同步继续执行；用 `info inferiors` 查看进程，`continue` 继续。

CPU Twin 不验证真实 NPU 调度、图捕获、资源生命周期或性能。Host 同步适配在 CPU launch 完成后返回，捕获调用由独立 NPU runner 验证。官方说明：[CPU 孪生调试](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0073.html)、[MIX/AIV 模式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_1211.html)。
