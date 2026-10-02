# BatchMatmulMaxSum 赛题契约

公开题面的开发用摘要，完整规则见[题目页面](https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum)。

## 数学语义

逻辑矩阵固定为 X1[B,M,K]、X2[B,K,N]：

```text
y[b] = sum_m(max_n(sum_k(X1[b,m,k] * X2[b,k,n])))
```

第 b 组只与第 b 组配对，无 batch broadcast。先沿 N 取最大值，再沿 M 求和，二者不可交换。不执行 L2 归一化，不接收 padding、有效长度或 mask。

## 数据与布局

| 张量 | 类型 | 连续 ND 存储形状 |
| --- | --- | --- |
| x1 | FP16 或 BF16 | transposeX1=false: [B,M,K]；true: [B,K,M] |
| x2 | 与 x1 相同 | transposeX2=false: [B,K,N]；true: [B,N,K] |
| y | FP32 | [B] |

两个 transpose 属性默认 false，仅声明存储布局，不改变逻辑矩阵与数学语义。四种组合都需支持。

- 1 ≤ B ≤ 64；1 ≤ M,N ≤ 8192。
- 32 ≤ K ≤ 8192，K 为 8 的倍数；两个输入的 B、K 相等。
- B×M×K ≤ 2^26，B×N×K ≤ 2^26。
- M、N 不保证对齐，必须处理尾块。无空张量，无 NaN/Inf，允许负数。
- 点积与两级归约使用 FP32 累加或等效精度。标准 golden 从实际量化后的输入存储值以 FP64 计算，最后转 FP32。
- 全负行的最大值不能初始化为 0；输出必须有限。不得修改输入，相同输入与属性多次执行结果须一致。

题面对 FP32 输出要求相对/绝对误差 1e-4。本地数值工具使用 abs(actual-golden) ≤ 1e-4 + 1e-4×abs(golden)，最终以赛事评测为准。

## 实现与提交

核心计算必须由 Ascend C 算子在昇腾 NPU 上完成。Host 只做元数据解析和启动；不能把数值计算移至 CPU，也不能用空 kernel 占位。

沿用工程提供的 run_kernel ABI：x1 与元数据、x2 与元数据、y 与元数据、availableCoreNum、stream、transposeX1、transposeX2。结构体布局和参数顺序以当前模板为准；当前类型编码 FP16=1、BF16=2、FP32=0。

赛事共 15 个测试点，全部精度通过后才计分。
