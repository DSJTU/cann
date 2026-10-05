# Ascend C 接口与同步要点

适用于 CANN 9.0、dav-2201。实验经验放 `.private/LESSONS.md`，此处记录当前实现与可复用验证方法。

- 数学顺序为完整 K 点积、N 最大值、M 求和。负行从负无穷初始化，尾块只归约有效元素；N 分区合并逐行 max，不能对 K 分区先 max。
- `GM_ADDR` 有 `__gm__` 地址空间限定。Matmul 转置模板、`SetTensorA/B` 与 Host tiling 一致，修改后须经目标 ASC 编译。
- 定义 `ASCENDC_CUBE_ONLY`，在 AIC 上直接 `Matmul::Init(&tiling,&pipe)`，不用 AIV Matmul 服务。`Iterate()` 和 `GetTensorC(GM,0,true)` 输出一块完整 K 点积；ND 尾块 GM 行距为有效列数，槽距为 `tileM*tileN`。
- MIX(1,2) 中 AIC 索引为 Cube 编号，AIV 索引除二为配对 Cube 编号，`GetSubBlockIdx()` 为 0/1。任务步长为 `workers/2`，全 AIV 归约步长为 `workers`。
- 两个 AIV 共享每个 Cube 的两槽。ready 使用 flag 4/5，free 使用 6/7；AIC 在两侧释放后复用槽，空尾半行和直接输出模式下的 AIV1 也须参与握手。消费者在 MTE2 完成 GM→UB 后释放槽，退出前生产者排空 free 信用。
- 分区结果在全 AIV `SyncAll<true>()` 后读取，所有 AIV 都参与屏障。跨核通知、队列释放和全核屏障用途不同，修改时必须验证真实设备时序与重复调用。
- `DataCopyExtParams.blockLen` 和 GM stride 为字节，UB stride 为 32 字节块；Gather 偏移为字节。输出精确搬运 4 字节，禁止相邻 batch 标量 GM 写回造成缓存线覆盖。
- 小矩阵直接声明静态 UB 地址，布局按 32 字节对齐；手工 HardEvent 保护搬运和 Vector 依赖。WholeReduce 的 repeat≤255，按 248 分段保持下一段对齐；行最大值存偶数 lane，最终位 mask 排除未写入的奇数 lane。
- scratch 按 context/stream 隔离。固定形状捕获、跨形状扩容和多 stream 是不同调用模式；通过某一模式不能声称其他模式通过。
- golden 从实际 FP16/BF16 存储值用 FP64 计算。CPU 模型、ASC 编译、设备数值、真实单 kernel profile 和赛事性能分别验证，结论必须关联源码哈希。
