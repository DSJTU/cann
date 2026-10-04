# Ascend C 开发要点

适用范围：CANN 9.0、dav-2201、kernel 直接调用工程。接口与硬件支持范围需核对目标 SDK。

## 算子数学与精度

- transpose 标志可能只声明存储布局；先固定逻辑矩阵公式，再分别推导四种 GM 地址。
- 最大值从有效首项初始化。全部值为负时，初始值 0 或将填充数据作为有效列都会错。
- golden 必须使用实际量化后的 FP16/BF16 存储值。先量化再以 FP64 计算，最后转 FP32；量化前随机数不是设备输入。
- FP32 归约次序会影响误差。近零抵消是重要测试，必要时使用补偿求和；跨核按固定次序汇总以保证重复确定性。
- M 求和补偿不能保护 K 点积。K=8192 时，先累加 2048 个 1，再累加 2048 个 2^-20，最后累加 2048 个 -1，普通 FP32 跨块累加会得到 0，实际存储值的 FP64 点积为 2^-9。跨 K 块补偿累加已在 CPU 模型和 A3 真机验证；它不消除块内归约及乘法自身的所有舍入误差。
- 输出尺寸和布局属于契约，只写有效输出。避免为了对齐扩大 GM 写入范围、改写输入、依赖旧输出或未初始化 UB。

## Host ABI 与地址空间

保留模板 `run_kernel` 参数顺序、元数据结构布局和调用约定。`.asc` 中的 launch 由 ASC 编译器处理，普通 C++ 编译成功不能证明 launch 生成、动态库加载和真实调用正确。

`GM_ADDR` 带 `__gm__` 地址空间限定，不能当成普通 `uint8_t*`。普通 `void*` 到 GM_ADDR 的 `reinterpret_cast` 可被 ASC 拒绝；设备侧同一地址空间内的转换是另一种情况。沿用经实际 SDK 验证的模板形式，先分清类型再修转换，不能机械改成 C 风格。

CPU 模型若把 `__gm__` 擦掉，会漏掉地址空间错误。用 [Host 类型检查](../solution/tests/check_host_types.py) 作为补充，最终仍要用目标 SDK 构建与执行。

## UB、DMA 与向量参数

Matmul 的输入类型 `MatmulType<POSITION, FORMAT, TYPE, ISTRANS>` 默认不支持转置。需要给 `SetTensorA/B` 传 true 时，先把对应类型的 `ISTRANS` 设为 true；仅设置 Host tiling 的转置参数不足以开启设备侧能力。开启后可在运行时传 true 或 false。小用例通过不能替代这一接口约束，仍需检查转置时的 L1 容量与重复分块调用。参见 [Matmul 模板参数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0623.html)。

| 项目 | 检查重点 |
| --- | --- |
| TPipe/TBuf | 缓冲字节总量，随最大 K 增长的空间，正确的 VECIN/VECOUT/VECCALC 位置 |
| DataCopyExtParams.blockLen | 字节数，不能用元素数代替 |
| GM stride | 相邻块末尾到下一块起点的字节间隙 |
| VECIN/VECOUT stride | 对应搬运接口按 32 字节块计；不要照搬 GM 单位 |
| padding | 按元素计，块对齐后 UB pitch 与逻辑长度可能不同 |
| Gather | 核对偏移单位；当前 FP32 偏移表按字节构造 |
| mask/repeat/stride | 各操作数分别检查有效范围、重复数、块与重复间跨度 |
| 尾块 | 搬运、索引、pitch、Gather 和归约共同验证，填充不参与有效最大值 |

单个 FP32 标量输出可用 DataCopyPad 精确搬运 4 字节。接口重载与产品支持不同，参数单位应对照 [DataCopyPad 官方表](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html) 和已安装 SDK。

`DataCopyPadExtParams` 的 left/right 虽按元素计，单侧填充对应的字节数各不超过 32。扩大 N 分块后不能直接将短尾块填到整个逻辑分块宽度；例如 32 列分块只剩 1 列 FP16，填 31 个元素会超限。按实际列数对齐到 16 元素得到合法物理 pitch，并让 Cast 与 Gather 使用同一 pitch。CPU 模型已加入该上限检查；这项遗漏曾造成目标 NPU 返回 507035。

## 核内与核间同步

完整 K 与分段补偿是不同的性能/精度取舍。旧版使用 128-K 面板、打包和 TwoSum，已修复若干跨面板抵消用例，但有明显搬运、IPC 和补偿开销。当前重建使用原生 full-K，不再把 128 作为通用累加长度；硬件 FP32 累加仍会丢失强抵消输入的低位。必须保留精度诊断并如实记录结果，不能用 FP64 CPU Matmul 的通过替代硬件舍入验证。

原生 Cube 的转置模板参数与 Host tiling 必须一致。`singleCoreM` 不超过 `baseM` 时，Iterate 沿 N 交回每个基本块，Sequential ND 尾块使用实际列数的紧密行距。当前以 per-worker GM 得分块暂存，DataCopyPad 搬入 UB 时对齐行距，WholeReduceMax 的 mask 只包括有效 N。同步的直接 UB 输出曾在同输入真机比较中更慢，已撤回；减少 GM 搬运并不自动减少 Cube/Vector 通信开销。

分派和块大小要以实际 kernel profile 比较，不能从理论吞吐量推断。原生完整 K 路径中，短/中 K 的宽 N 块可以摊薄通信成本；长 K 小 M/N 则可能因为独立任务过少而闲置核。减少 M/N 块可增加并行度，也会增加请求和尾块成本，须在相同输入上测量。小矩阵的 Vector 路径一次缓存输入，用 K 并行乘法与树形归约避免 Cube server 的固定开销。

M 求和的树形归约仍可能有抵消误差。当前 Finish 先计算普通和与绝对值和；结果绝对值小于绝对值和的 1% 时，按固定行序以 TwoSum 重算，跨两段 M 保留同一高/低状态。这个判断是性能与数值的工程取舍，不修复 K 点积内部丢失的低位。

历史 IterateBatch 实验的约束仍值得保留：NORMAL `BATCH_LARGE_THAN_L1` 可能把偶数批次拆成奇数 L1 主块而漏掉主块末项，A3 曾在多种面板数上复现；补零到 2 的幂可以避开已测问题。非连续 GM 输出需使用完整 singleCoreN 行距，`matrixStrideC` 为保留参数。当前主路径不使用面板批量，不能继续按旧分派门槛或旧 kernel 名称验收。

源码顺序不代表 Scalar、Vector、MTE2、MTE3 已完成。按生产者→消费者选 HardEvent，通过 TPipe 获取事件 ID，配对 SetFlag/WaitFlag；基线中的常见关系：

| 依赖 | 事件 |
| --- | --- |
| 输入搬运→Vector | MTE2_V |
| Vector→Scalar | V_S |
| Scalar→Vector | S_V |
| Scalar→输出搬运 | S_MTE3 |
| 输入搬运→Scalar | MTE2_S |
| Scalar 读完→下一次搬入 | S_MTE2 |

先证明依赖再删屏障。CPU 中同步执行的算子模型无法发现所有真实异步流水问题。

Matmul 的 `GetTensorC<true>` 并不替代消费者到 Scalar 的完成依赖。直接 UB 输出实现需在复用 C 或 `End()` 前等待 V_S，让归约与最大值合并真正完成。缺失这项依赖曾使 M=8192 的重复调用改变 78 个行最大值；补上后相同数据的重复、大型回归与图捕获全部通过。

硬件 SyncAll 要求所有参与核执行相同数量的屏障，逻辑核数不能超过可同时驻留的物理核数。带屏障的任务不应分时过量调度或放进不同核迭代次数不一致的循环。纯 Vector 硬同步使用 `__mix__(0,1)`；可能存在多 stream 并发时，batch 调度 `__schedmode__(1)` 有助于避免核间等待造成的死锁。参见 [SyncAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md)、[调度修饰符](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html)。较新文档不能替代目标版本验证。

混合核 `__mix__(1,2)` 中，`GetBlockNum()` 返回 launch 配置的逻辑 block 数；AIV 的 `GetBlockIdx()` 范围却包含每个 block 的两个 Vector worker。任务循环步长应为 `2*GetBlockNum()`（或按当前核的 task ratio 计算），纯 Vector 核仍用原步长。已核对 CANN 9.0 的通用 `GetBlockNum()` 和 dav_c220 的 `GetBlockIdxImpl()` 实现。忽略配比会让不同 AIV 重复写同一行区间；小规模数值通过不能证明任务划分互不重叠。

小 batch 只按 batch 分核可能闲置大部分计算核。沿 M 分配独立行时，证明分区完整、无交叠、无空段，且最终归约顺序明确。少量标量输出可作为轮流传递分区和的 GM 位置，配合全核屏障与固定顺序归约；这适用于当前算子，不是通用并行归约方案。

## 资源生命周期与调用方式

图捕获会延后执行。捕获期间不能对相关 stream 做普通同步；被图引用的设备内存需存活到图销毁，不能随 Host 函数返回立即释放。需要释放回调时区分每图登记与每次调用资源，验证同图多次调用，检查 API 返回值。

只检查调用方最后的 stream sync 会漏掉 Host 内部被忽略的错误。动态库能 dlopen 不代表 ABI 实际调用有效，需由独立 C++ 调用方执行数值用例。

## 验证阶梯与诊断

赛事每次迭代必须恰好启动一个 kernel。两阶段 Cube→Vector 即使精度、捕获和资源生命周期检查通过，仍会被 profiling 规则拒绝。每次关键检查点需核对真实设备 launch 数量；不能仅把两次 kernel 时间相加当作符合提交约束。融合 MIX 核中，计算阶段结束后每个 AIV（含无任务 worker）都恰好执行一次 SyncAll，再读取独立行最大值。使用 batch 调度防止多 stream 交叠造成全核汇合死锁，且归约 worker 步长为两倍逻辑 block 数。计算与最终归约的 TPipe 分阶段销毁/初始化以复用 UB，仍需真实 SDK 和设备验证。

| 检查 | 能证明的范围 |
| --- | --- |
| CPU 模型 + ASan/UBSan | 控制流、索引、边界、初始化、模型下数值与确定性 |
| 官方 CPU Twin | 同一计算体在 tikicpulib 中执行，可用 GDB 调试 AIV/MIX；仍需单独验证实际 NPU 调度、图捕获和舍入 |
| Clang 地址空间检查 | Host 代码的部分类型约束 |
| 目标 ASC 编译/链接 | SDK、launch 生成与符号兼容 |
| 官方 camodel | 仿真的设备指令与执行路径，不能代替真机性能 |
| NPU 真机 | 已测输入及调用模式的真实执行、精度和运行行为 |
| 赛事评测 | 实际平台固定测试集的结果，不代表所有合法输入或性能最优 |

测试至少覆盖两种类型、四种布局、负值、零值、近零抵消、K/N 尾块、分区余数、不同核数、相邻 batch、输入不变、输出保护区和重复调用。对同步、ABI 或资源管理有改动时，加测首次无预热捕获、同图连续调用、多 stream 和独立动态库调用。

Compile Error 先查类型/编译/链接；Runtime Error 再分启动、设备异常与超时。Skipped 不是独立精度失败。定位原因需要构建日志、ACL 返回值和设备日志，不能仅凭结果标签判断。

上板产物不能仅靠 LD_LIBRARY_PATH/软链接变成仿真产物；camodel 需对应构建模式与 runtime/driver。镜像名称和资源类型也不证明已具备相应设备能力，先核对工具链、实际设备和支持的编译目标。

不完整输出不能算通过，独立核验应检查数量。仿真可能很慢，区分仿真进程上限与提交 kernel 超时。Host wall_us 包含启动、同步及可能的资源管理，不能和平台纯 kernel 时间直接比较；性能结论需要相同输入、设备和计时口径。
