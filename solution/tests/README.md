# 用例与性能筛选

隐藏的 15 个赛事测试点没有已知形状映射。本目录的输入均为自建诊断用例，不能按赛事截图编号猜形状，也不把本地加速比称为赛事分数。

## 正确性

从仓库根运行：

```bash
mkdir -p .private/runtime/tmp
TMPDIR="$PWD/.private/runtime/tmp" python3 solution/tests/test_cpu.py
TMPDIR="$PWD/.private/runtime/tmp" python3 solution/tests/test_validation.py
```

CPU 模型默认包含 `robustness`：各行最大值出现在不同列、64 个不同 batch、最后有效列为负的最大值、长 K 尾部、4095/4096/4097 行归约边界。结合已有的两种精度、四种布局、负数、抵消、大小数、M/N 尾块、重复性、输入不变和输出保护区检查。`performance` 套件在 CPU 上仍然只验证数值。

工具回归会故意提供缺失/重复计时、旧 profile、错误输出、改写的数据、错配构建记录及噪声收益，检查其被拒绝或被标为无法判断；这些日志是测试工具的合成输入，不代表真机结果。

在目标服务器重新配置和构建当前测试工程后运行：

```bash
cmake -S solution/tests -B .private/runtime/npu/build \
  -DNPU_ARCH=dav-2201 -DBMMMS_TRACE_RUNTIME=ON
cmake --build .private/runtime/npu/build -j4
python3 solution/tests/run_npu_checks.py --suites baseline robustness \
  --modes ordinary --runs .private/runtime/npu/check-new
```

只在前一步退出 0 后执行下一步。`--runs` 必须是全新目录。runner 的成功链接会生成 `.build.json`，记录源码、runner、构建配置、二进制和编译器哈希、架构与编译选项；源码变化或替换旧二进制后必须重建。普通调用通过不代表捕获、多流等其他模式通过。真实每次单 kernel 数量仍通过 msprof 单独检查。

检查数学契约和启动数量，不强制复制参考方案的 Cube/Vector 分派阈值或 scratch 分配数量。合法的分派优化不能被旧阈值模型判错；报告保存实际观测的 kernel 名称。新增入口需要更新 profile 名称识别，并重新核对单 kernel 规则。

## 几何覆盖与保留验证集

性能集定义在 [case_catalog.py](case_catalog.py)。每套包含 9 类 × 4 个形状 = **36 个几何形状**，展开 FP16/BF16 × 四种存储布局为 **288 个调用配置**。两套共 72 个互不重复的形状：

| 分类 | 目的 |
| --- | --- |
| dot | 极小点积的启动成本、K=32/64 与动态长 K |
| small | 小矩阵、不同 batch、K 尾部及转置搬运 |
| dispatch-boundary | 小矩阵分派两侧，防止只调一个阈值 |
| skinny | M=1、N=1 和低行/列利用率 |
| batch | 不同 batch 并行度、任务不足与任务复用 |
| rectangular | 长宽交换、大 N 扫描与大 M 归约 |
| balanced | 中大矩阵的计算与 tile 调度 |
| long-k | K=1024…8192、面板边界与 8 元素尾部 |
| large-reduction | 大 M、多块归约与非对齐尾块 |

- `benchmark-tune` 用于日常调优。
- `benchmark-holdout` 只在候选方案定型后检查；不要根据它反复改阈值。反复调过的验证集就不再有独立验证价值。
- `robustness` 检查容易误算的结构化数据，不混入性能统计。
- 旧 `benchmark` 保留为快速诊断集，不能用于宣布总体提升。

形状与数据模式决定独立随机种子；新增、删减或重排用例不会改变其他用例的输入。四种布局共享同一逻辑矩阵，精度变体只改变实际存储量化。`--seed` 可更换输入，不能代替未见过的几何形状。

FP64 golden 从实际 FP16/BF16 存储值计算。生成器同时写入 `.bin`、`.golden.bin`、`.json` 和 `.manifest.json`，验证时核对输入、golden 和元数据哈希。默认调优/验证集分别约 132/174 MiB 输入、101/114 亿次数学浮点运算；适合关键性能检查点，不作为每次小改动的前置条件。大规模压力测试仍可单独使用 `stress`。

## 两个版本交替对比

为基线与候选使用**同一份当前测试工程**，分别放入两个版本的 `kernel.asc` 后构建 runner；只有算子源码不同。使用同设备、SDK、架构及编译选项，测试期间不运行其他 NPU 任务。仅复制旧 executable 不足以建立可信对照。

示例中两侧构建目录位于 `.private/runtime/`。以下命令假定已得到两个完整构建及各自的 `.build.json`：

```bash
python3 solution/tests/run_benchmark.py \
  --baseline .private/runtime/baseline-build/bmmms_npu_runner \
  --candidate .private/runtime/candidate-build/bmmms_npu_runner \
  --suite benchmark-tune --rounds 4 \
  --runs .private/runtime/bench-tune-new
```

候选定型后，换全新目录检查保留集：

```bash
python3 solution/tests/run_benchmark.py \
  --baseline .private/runtime/baseline-build/bmmms_npu_runner \
  --candidate .private/runtime/candidate-build/bmmms_npu_runner \
  --suite benchmark-holdout --rounds 4 \
  --runs .private/runtime/bench-holdout-new
```

脚本以 AB、BA、AB、BA 顺序执行四轮配对，降低运行顺序带来的温度/频率漂移影响。每个进程每例预热 2 次、测量 10 次；每轮两侧均重新检查输出、重复性、输入不变、保护区、设备与核数、实际调用配置和真实单 kernel 数量。数值失败、数据变化、过期二进制或不完整 profile 会使整个流程失败。

每套四轮共采集 27648 次启动，**不是 27648 个独立性能样本**。每例每轮取 10 次纯 kernel 耗时中位数，统计不把同一进程内的十次重复当成十个独立轮次。EVENT 和 wall 时间用于诊断提交/同步成本；收益筛选使用 msprof `Task Duration(us)`，不混用计时口径。

## 解读结果

`results.json` 保存构建身份、数据哈希、两侧执行顺序、设备信息、数值报告、原始 kernel samples 和对比结果。`workflow_passed=true` 表示测量流程有效，**不表示候选更快**。

- `balanced_candidate_over_baseline` 小于 1 表示此集合上更快，按分类、几何形状、精度/布局分别等权计算几何平均；避免某一类大量重复用例支配汇总。它不是官方计分公式。
- `groups` 同时列出分类、精度和布局收益；`geometry_comparison` 按形状列出退步；`worst_regressions` 展示最差配置。不得只引用一个总体中位数或极端最小耗时。
- `balanced_ratio_bootstrap_95pct` 按整轮重采样，只描述此固定集合的运行噪声，不代表隐藏形状的可信区间。至少四轮才给筛选判断；接近噪声边界时增加轮次，并先复测基线对基线。
- `screening_status` 为 `insufficient_rounds`、`inconclusive_noise`、`slower_on_this_corpus`、`mixed_tradeoff` 或 `promising_on_this_corpus`。分类退步超过 3% 或某配置退步超过 10% 会显示取舍；这些阈值是诊断提示，不是赛事规则或自动晋级条件。

先用调优集找收益，随后用保留集检查几何泛化，再提交赛事验证。自建集变快而赛事分数不变时，重新审视分派、工作量与计时口径，不能用“用例全过”或合成平均收益替代赛事提升。
