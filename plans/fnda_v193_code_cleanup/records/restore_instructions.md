# S3 恢复说明

[总体计划](../MASTER_PLAN.md) · [基线报告](baseline.md) · [备份清单](backup_manifest.json)

源快照：`D:\code11\IRI01\IRI03\INR1-1-run65_backup_v193_20261009_S3`。其中 project 保存项目文件；外部来源与实际副本路径以 manifest 的 source_roots 和 copied[].snapshot 为准。

1. 恢复前停止会写入目标文件的任务，将待恢复文件的现状保存到另一个独立目录，保留失败现场。
2. 从 backup_manifest.json 的 copied 数组按完整 path 查找原路径，拼接备份根目录与 snapshot 得到副本。恢复前检查副本 SHA-256 与 sha256 相同。
3. 先恢复到新建的独立目录，保持相对结构，逐文件验证；不要直接整体覆盖工作区。4 个核心文件的恢复抽检已通过，记录见 restore_sample。
4. 仅在明确需要回滚相应批次时，将核对后的文件恢复到其原路径；外部代码必须按原路径映射处理，不混入活动 Python 搜索路径。
5. 重新验证该批受影响的导入、固定权重与行为对照，并记录结果。Git bundle 只包含 Git 历史，未提交内容以文件快照和补丁为准。

registered_in_place 的 30,202 个文件没有副本，必须原位保留，不得把登记清单当作灾备。300 个 unreadable 路径未经备份，按用户决定原位保护且不准迁出。任何原位文件如将被覆盖/迁出，须先另行可读备份并验哈希。

本阶段最终计划、报告及基线输出另存 stage_s3_supplement；补充包不改变原始快照和原 backup_manifest.json 的语义，其自身哈希清单独立保存。
