# 开发说明

修改前阅读 cann_problem.md、solution/kernel.asc 和相关测试。Ascend C 接口与同步注意事项见 docs/lessons.md。

- 核心计算在 NPU 上完成；CPU golden 仅用于验证。
- 保持 run_kernel ABI、输入不变与重复确定性。
- 修改搬运、地址空间、精度或同步时，验证对应边界与调用模式。
- 本地检查：python3 solution/tests/test_cpu.py。它不能替代目标 ASC 编译和真机验证。
