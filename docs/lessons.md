# Ascend C 接口与同步要点

适用于 CANN 9.0、dav-2201 和直接调用 `run_kernel` 的工程。接口与硬件支持以目标 SDK 为准；构建和测试命令见 [构建说明](../solution/README.md)。

## 数学、布局与精度

- transpose 只声明存储布局，先固定逻辑公式，再推导四种 GM 地址。
- 全负行的最大值不能从 0 初始化；N 填充列不能参与最大值，M 填充行不能参与求和。
- golden 从实际量化后的 FP16/BF16 存储值计算 FP64 结果，再转 FP32。
- K 点积与 M 求和是两个独立的误差来源。完整 K 的 FP32 累加可能丢失强抵消输入的低位，后续求和补偿不能恢复它们。保留不同量级和近零抵消诊断。
- 输出只写有效范围；不得改写输入、依赖旧输出或未初始化 UB。跨核归约使用固定次序保证确定性。

## ABI 与 Matmul

保留 `run_kernel` 参数顺序、元数据结构布局和调用约定。`GM_ADDR` 带 `__gm__` 地址空间限定，不能当成普通 `uint8_t*`；普通 C++ 编译无法证明 ASC launch、地址空间转换和动态库调用正确。

`MatmulType<POSITION, FORMAT, TYPE, ISTRANS>` 默认不启用转置。传给 `SetTensorA/B` 的转置能力必须在模板中启用，Host tiling 参数也须一致，参见 [模板参数](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0623.html)。

当 `singleCoreM ≤ baseM` 时，`Iterate` 沿 N 交回基本块。Sequential ND 输出的尾块使用实际列数作为紧密行距；搬入 UB 时再对齐物理行距，归约 mask 只包括有效列。

## UB 与搬运

| 参数 | 单位或约束 |
| --- | --- |
| TPipe/TBuf | 按字节核算总量与缓冲位置，包含 Matmul 占用 |
| DataCopyExtParams.blockLen | 字节 |
| GM stride | 块末尾至下一块起点的字节间隙 |
| VECIN/VECOUT stride | 对应接口按 32 字节块计 |
| padding | 元素；单侧填充对应字节数不超过 32 |
| FP32 Gather 偏移 | 字节；与搬运后的物理 pitch 一致 |
| mask/repeat/stride | 分别验证操作数范围与块、重复间跨度 |

短尾块应按搬运对齐要求计算 pitch，不能填充至任意逻辑分块宽度。单个 FP32 输出可用 `DataCopyPad` 精确搬运 4 字节，参见 [官方参数表](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0265.html)。

## 同步与任务划分

源码顺序不表示 Scalar、Vector、MTE2、MTE3 已完成。按生产者到消费者选择 HardEvent，通过 TPipe 获取事件 ID，并配对 SetFlag/WaitFlag：

| 依赖 | 事件 |
| --- | --- |
| 输入搬运→Vector | MTE2_V |
| Vector→Scalar | V_S |
| Scalar→Vector | S_V |
| Scalar→输出搬运 | S_MTE3 |
| 输入搬运→Scalar | MTE2_S |
| Scalar 读完→下一次搬入 | S_MTE2 |

`GetTensorC<true>` 不代替后续消费者的完成依赖。GM 得分槽被下一次 Cube 请求复用前，必须等待读取该槽的 MTE2 完成；Scalar 发起请求时仅有 MTE2_V 不足。Vector 归约供 Scalar 读取时须等待 V_S。

`__mix__(1,2)` 的逻辑 block 有两个 AIV worker，AIV 任务步长是 `2*GetBlockNum()`；纯 Vector 核使用原步长。证明每个分区完整、无交叠，所有 worker（包括无任务 worker）恰好执行一次最终屏障。

`SyncAll` 要求参与核执行相同数量的屏障，且逻辑核数不超过可同时驻留的物理核数。多 stream 并发需核对调度，参见 [SyncAll](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/ascendcopapi/docs/en/api/SIMD-API/basic_api/sync_control/inter_core_sync/SyncAll.md) 和 [调度修饰符](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10053.html)。较新文档仍需目标版本验证。

## 生命周期与验证范围

图捕获延后执行，被图引用的内存必须存活至图销毁。缓存按 context/stream 区分，扩容不能释放旧图的分配；释放缓存前完成同步，并在 stream/context 销毁前清理。检查 Host 内部 ACL 返回值，独立 C++ 调用方实际验证动态库 ABI。

| 检查 | 范围 |
| --- | --- |
| CPU 模型 + ASan/UBSan | 控制流、索引、初始化、边界与模型下数值 |
| Clang 地址空间检查 | Host 的部分类型约束 |
| 官方 CPU Twin | 官方 CPU 库执行同一计算体及支持的同步/搬运检查 |
| 目标 ASC 编译/链接 | SDK、launch 生成与符号兼容 |
| 官方 camodel | 仿真设备指令与路径，不能代替真机性能 |
| NPU 真机及 profile | 已测输入、调用模式的实际精度、调度与启动数量 |

测试覆盖两种类型、四种布局、负值、零值、抵消、M/N/K 尾块、分区余数、输入不变、保护区和重复调用。修改同步、ABI 或资源管理时，加测无预热捕获、同图连续调用、多 stream 和动态库调用。所有分派均须每次恰好一个实际 device kernel，数值通过不能替代 profile 验收。

CPU Matmul 的 FP64 模型不模拟硬件舍入或异步流水。检查输出数量，未完成或缺失输出不算通过。Host wall time、stream-event 时间与纯 kernel 时间口径不同；比较性能需相同输入、设备与计时方式。
