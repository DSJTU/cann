# 文档索引

当前实现与验收以 [单次启动检查点](single-kernel.md) 和 [原始设备报告](single-kernel-data.json) 为准。源码 `54cc8bd` 已经设备回归、用户赛事验证并确认提升。开发接口与同步约束见 [lessons.md](lessons.md)，构建和最小验证入口见 [solution/README.md](../solution/README.md)。

## 历史记录

历史报告和原始数据统一位于 [history/](history/)。其中“当前”“候选”“下一步”均指记录时点，不能继承为最新源码的验收结论。

| 阶段 | 记录 | 状态 |
| --- | --- | --- |
| Vector 正确性基线 | [validation](history/validation.md) | 历史通过版本 |
| Vector 性能优化 | [performance](history/performance.md) | 历史通过版本 |
| 短 K Cube 原型 | [cube-performance](history/cube-performance.md) | 历史两阶段实现 |
| Finish 补偿树 | [finish-performance](history/finish-performance.md) | 设备收益未转化为明显赛事收益 |
| 长 K Cube 设计 | [long-k-performance](history/long-k-performance.md) | 两次启动被赛事规则拒绝；设计反例保留 |

历史目录中的 JSON 保留原始内容。详细赛事成绩保存在本地 `.private/SCORES.md`，随日期、提交和源码哈希查询，不随源码上传。
