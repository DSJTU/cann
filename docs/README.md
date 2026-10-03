# 文档索引

当前实现、分派和本地检查见 [solution/README.md](../solution/README.md)。接口与同步约束见 [lessons.md](lessons.md)。

本分支仍在开发。当前候选扩大短 K Cube 覆盖，并加入 N 分区；长 K 使用面板批量 Cube。当前源码尚未赛事评测或设备回归。`main` 上完成设备回归并经用户确认提升的版本是 `54cc8bd`。逐点耗时不在共用仓库。

## 历史记录

下列记录在当前融合实现之前就在共用仓库中。其中“当前”“候选”“下一步”均指记录时点。

| 阶段 | 记录 | 状态 |
| --- | --- | --- |
| Vector 正确性基线 | [validation](history/validation.md) | 历史通过版本 |
| Vector 性能优化 | [performance](history/performance.md) | 历史通过版本 |
| 短 K Cube 原型 | [cube-performance](history/cube-performance.md) | 历史两阶段实现 |

历史目录中的 JSON 保留当时的原始内容。详细赛事成绩保存在本地 `.private/SCORES.md`，不随源码上传。
