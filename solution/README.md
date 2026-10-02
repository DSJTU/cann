# BatchMatmulMaxSum

当前实现已通过赛事 15/15。提交复制 [kernel.asc](kernel.asc) 全部内容，保留模板 run_kernel ABI；数值计算全部在 NPU。

小任务按 batch 分核，大任务沿 M 分区。各核完成 FP32 点积、N 最大值、分区补偿求和，再通过硬件屏障与输出位置按固定顺序汇总。一次 kernel 启动，无额外 GM 工作内存或原子累加。

在仓库根运行本地检查，依赖 g++、clang++、NumPy：

```bash
python3 solution/tests/test_cpu.py
```

280 组覆盖两种类型、四种布局、负值、抵消、尾块、跨 batch、并行汇总、输入不变、输出保护区与确定性；启用 ASan/UBSan。类型模型与 CPU 模型不能代替目标 SDK 编译和真机验证。

NPU 验证工具位于 [tests/](tests/npu_runner.asc)：独立 FP64 golden、普通/冷捕获/同图连续调用/双 stream、普通 C++ 动态库调用。构建前加载目标 CANN SDK 环境。

```bash
cd solution
cmake -S tests -B build-npu -DNPU_ARCH=dav-2201 \
  -DBMMMS_TRACE_RUNTIME=ON -DBMMMS_BUILD_SHARED_TEST=ON
cmake --build build-npu -j4
python3 tests/npu_data.py --suite correctness --prefix lab-data/correctness
./build-npu/bmmms_npu_runner lab-data/correctness.bin lab-data/result.out.bin
python3 tests/npu_data.py --prefix lab-data/correctness --verify lab-data/result.out.bin
```

搬运、同步与精度注意事项见 [Ascend C 开发要点](../docs/lessons.md)。
