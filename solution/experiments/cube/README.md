# Cube 分块原型

历史两次启动原型，违反当前赛事每次迭代一个 kernel 的约束。当前实现见 [构建说明](../../README.md)。

独立原型以 A3 / Ascend910_9362 / CANN 9.0 / dav-2201 为目标，仅接受 K≤128。主实现的短 K Cube 源于这个版本。当前分支的长 K 走面板批量 Cube；小矩阵和窄矩阵仍用 Vector。

混合核用 Matmul 的 GM ND 输入、FP32 VECIN 输出计算 32×64 得分块，立即沿 N 归约，只把行最大值写入 GM；第二个 Vector kernel 按原始 M 顺序补偿求和。用户工作内存为 `B*M*4` 字节，另需 Matmul 系统工作内存，不保存完整得分矩阵。

关键约束：

- `__mix__(1,2)` 的 AIV 任务步长为 `2*GetBlockNum()`。
- A/B 的 `MatmulType` 必须开启转置能力，再按实际布局传 `SetTensorA/B`。
- 顺序 ND 输出按有效尾块列数紧密排列；非 8 对齐列宽先 Gather，再归约。
- 每次输出的 Vector 消费完成后使用 V_S，再结束 Matmul 上下文或复用输出缓冲。
- 普通调用等待 stream 完成再释放内存；图捕获每次调用持有独立缓冲，同一个图只登记一次销毁回调。同图多次调用与双图提交分别验证。

2026-10-03 实测版本 SHA256：`26db80dc1981ec3fc0332e6bc875168c52d665fcc751937ca72581839f78a8af`。目标编译、普通模式 72 个分块用例、48 个短 K 精度用例及两次完整 24 个大型用例全部通过；大型 benchmark/profile 也均通过。冷捕获、同图三次调用、双 stream 的图重放、独立 C++ 动态库冷捕获，分别在 72/48/248/24 例集合全部通过。重复确定性、输入不变、输出保护区均由 runner 检查；数值由实际量化输入的独立 FP64 golden 核验。

直接调用 runner 的三种捕获模式共分配/释放 5,488 个缓冲，登记 1,568 次图销毁回调，392 次普通预热同步，无内部 ACL 错误。独立动态库模式核验数值与调用行为，没有上述内部计数。双 stream 测试先提交两个图重放再同步，不以此声称已测得硬件执行重叠。

```bash
source /home/developer/Ascend/ascend-toolkit/set_env.sh
cmake -S solution/experiments/cube -B /tmp/bmmms-build-cube \
  -DBMMMS_TRACE_RUNTIME=ON -DBMMMS_BUILD_SHARED_TEST=ON
cmake --build /tmp/bmmms-build-cube -j4
python3 solution/experiments/cube/data.py /tmp/bmmms-cube-data --suite short-validation
/tmp/bmmms-build-cube/bmmms_cube_probe /tmp/bmmms-cube-data.bin /tmp/bmmms-cube-output.bin --capture-cold
python3 solution/tests/npu_data.py --prefix /tmp/bmmms-cube-data --verify /tmp/bmmms-cube-output.bin
```

生成器另支持默认 `tiles`（72）、`precision`（48）、`large`（24）集合。性能与主实现验证见 [Cube 性能记录](../../../docs/history/cube-performance.md)。接口参考 [CANN 9.0 GetTensorC](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0639.html)，并已核对安装 SDK 的输出布局。
