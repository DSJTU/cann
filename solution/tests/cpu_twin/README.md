# 官方 CPU Twin

在本机使用 CANN 9.0 的 `tikicpulib` 执行 Ascend C 计算体，无需 NPU 或驱动。它与 `test_cpu.py` 的自写同步模型是两条独立验证路径。Host 的 launch、共享 GM 分配和 stream/capture 查询由测试适配层处理；算子计算体直接从当前 `solution/kernel.asc` 生成，比赛提交文件不改动。生成器遇到新的 launch 结构会拒绝构建，生成清单记录 kernel SHA256。

需要 x86_64 或 ARM 对应的官方 CANN Toolkit、C++17 编译器、CMake、GDB，以及有 NumPy 的 Python 环境。当前本机工具链已安装在仓库的 `.private/toolchains/cann-9.0`。其他电脑可设置 `BMMMS_CANN_ROOT`（包含 `set_env.sh` 的 SDK 根）和 `BMMMS_TWIN_PYTHON`（Python 可执行文件）；`CXX` 可指定编译器。官方说明见 [CPU 孪生调试](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0073.html) 与 [MIX/AIV 模式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_1211.html)。

从仓库根运行：

```bash
bash solution/tests/cpu_twin/run.sh --suite all
bash solution/tests/cpu_twin/run.sh --suite long-precision
bash solution/tests/cpu_twin/run.sh --suite long-batches
bash solution/tests/cpu_twin/run.sh --suite long-cube --case 8 --gdb
```

`smoke` 有 16 组 Vector，`short-cube` 有 8 组短 K Cube，`long-cube` 有 16 组长 K Cube（完整 K、M/N 尾部）；`small-matrix` 有 24 组小矩阵边界与多 batch 复用；`all` 共 64 组。均包含 FP16/BF16 和四种转置，显式使用 1 个逻辑核，MIX 模式由官方库启动 1 个 AIC 与 2 个 AIV 进程。每组重复两次，检查输入不变、输出保护区、确定性和一个 CPU launch，然后使用独立 FP64 golden 核验完整输出。`long-precision` 另外提供 32 组大小量级抵消诊断，包含正负近零结果、M/N/K 尾块、多个窗口、两种类型、四种布局和 1/2 个逻辑核；不包含在 `all` 中；当前原生完整 K 路径仍有已知失败。产物和日志写到 `.private/cpu-twin`。

`long-batches` 保留 8 组历史 L1 批次拆分输入，覆盖 10/14 个 K 面板、128/256 列窗口、两种类型与 NN/TT 布局。比较历史源码时，可在独立 CMake 构建目录传 `-DKERNEL_SOURCE=/绝对路径/kernel.asc`；生成清单仍记录该源码的 SHA256，当前默认构建继续使用工作区 kernel。

在 GDB 中对 `kernel.asc` 设置断点。例如执行 `break kernel.asc:行号`，然后执行 `run`、`info args`、`next` 或 `print`。调试入口默认跟随 fork 子进程，保留其他进程并启用 `schedule-multiple`，以便核间同步继续执行；不同核命中断点或退出时可能多次停住，使用 `info inferiors` 查看进程，`inferior 1` 返回 Host，`continue` 继续。断点行号随版本变化；在 `NativeCube::Process` 或 `SmallMatrix::Process` 对应源码行设断点。

CPU Twin 验证算子计算逻辑及官方 CPU 库支持的同步/搬运检查。它不验证真实 NPU 调度、图捕获、资源生命周期、赛事 launch profile 或真机性能，也不能把已测精度推广到所有硬件舍入和隐藏输入。Host 的同步适配在 CPU launch 完成后才返回；图捕获注册明确报错。比赛版本仍按既有设备检查和赛事评测流程验收。
