# Ascend C 接口与同步要点

适用于 CANN 9.0、dav-2201。实验结论放 `.private/LESSONS.md`，此处只记录可复用的接口约束。

- 数学顺序是完整 K 点积、N 最大值、M 求和；转置只影响存储地址。全负行从负无穷初始化，归约只含有效元素。
- `MatmulType` 的转置模板、`SetTensorA/B` 与 Host tiling 一致。`GM_ADDR` 有 `__gm__` 地址空间限定，须经目标 ASC 编译验证。
- 当前异步 `Iterate<false>()` 发起整个 N 窗口；无参数 `GetTensorC<false>()` 逐块返回 GM workspace 槽。CType 的 VECIN 标记不表示直接返回 UB。槽距为 `baseM*baseN`，ND 尾块内部行距是有效列数。
- `DataCopyExtParams.blockLen` 和 GM stride 是字节；UB stride 通常是 32 字节块。FP32 搬入行距按 8 元素对齐，mask 排除填充列。Gather 偏移为字节。
- 独立输出槽允许 Cube/Vector 重叠；窗口复用前等待 MTE2_S，再 `mm.End()`。队列负责搬入与 Vector 的依赖，指令间按需要使用 PIPE_V。
- MIX(1,2) 有两个 AIV worker，步长为 `2*GetBlockNum()`。需要最终归约时所有 AIV（含闲置者）执行同一次 `SyncAll<true>()`，逻辑核数不能超过物理并发能力。
- 同一 stream 可复用 scratch；固定形状捕获须能正常重放。比赛优化不保留已扩容旧 scratch 的旧图。
- golden 从实际 FP16/BF16 存储值以 FP64 计算。CPU 数学模型验证布局与边界，不模拟异步时序、硬件舍入或耗时。
- 目标编译、设备数值、实际 kernel 启动数与性能分别验证。性能比较使用同设备同输入的 msprof kernel duration，不能以 Host wall time 代替。
