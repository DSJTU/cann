# Ascend C 接口与同步要点

适用于 CANN 9.0、dav-2201。实验结论放 `.private/LESSONS.md`，此处只记录可复用的接口约束。

- 数学顺序是完整 K 点积、N 最大值、M 求和；转置只影响存储地址。全负行从负无穷初始化，归约只含有效元素。
- `GM_ADDR` 有 `__gm__` 地址空间限定。`MatmulType` 的转置模板、`SetTensorA/B` 与 Host tiling 一致；修改须经目标 ASC 编译验证。
- SDK 的 `Iterate<false>()` 发起 N 窗口，无参数 `GetTensorC<false>()` 逐块返回 GM workspace 槽。VECIN 标记不表示直接返回 UB。槽距为 `baseM*baseN`，SDK ND 尾块内部行距是有效列数。
- 手写路径的 L1 为 NZ，L0A 为 ZZ、L0B 为 NZ。原始 16 位重排可用于两种输入精度，不能把 BF16 数值按 FP16 转换。尾部写入完整零填充面板；Mmad 的 K 长度与 L0 布局保持一致。
- 当前设备的 CO1→UB Fixpipe 接口不支持此用法，输出经 GM 独立槽再进入 UB。手写槽的 GM 行距是 tileN，Fixpipe 的 srcStride 是 tileM；搬入 UB 后行距按 8 个 FP32 对齐。
- `DataCopyExtParams.blockLen` 和 GM stride 是字节；UB stride 通常是 32 字节块。Gather 偏移为字节，归约 mask 排除填充列，最终输出精确搬运 4 字节。
- SDK 窗口复用前等待 MTE2_S，再 `mm.End()`。手写路径保护 MTE2→MTE1、MTE1→M、M→MTE1、MTE1→MTE2 与 M/FIX 依赖；不得提前覆盖旧 L0 操作数或 C0。
- 手写 MIX(1,1) 的两槽 ready 使用 flag 6/7，消费者完成 GM→UB 后释放 flag 8/9；重排完成使用 flag 10。SyncAll/KFC 的 11–15 保留通道不能用作这些槽的握手。
- 手写预处理和分区结果写回后，先完成 MTE3→S 依赖，再做全核屏障；局部队列释放或异步跨核通知不能替代全部 GM 写回完成。预处理的 TPipe 销毁及 UB 复用必须在该等待后。
- MIX(1,2) 有两个 AIV worker，步长为 `2*GetBlockNum()`。全核归约或预处理屏障包括所有 AIV，含闲置者；逻辑核数不能超过物理并发能力。
- 同一 stream 可复用 scratch；固定形状捕获须能重放。比赛实现不保留已扩容旧 scratch 的旧图。不同 stream 使用独立 scratch。
- golden 从实际 FP16/BF16 存储值以 FP64 计算。CPU 数学模型验证布局与边界，不模拟异步时序、硬件舍入或耗时。
- 目标编译、设备数值、实际 kernel 启动数与性能分别验证。性能比较使用同设备同输入的 msprof kernel duration，不能以 Host wall time、模型请求数量或自建极端形状代替赛事成绩。
