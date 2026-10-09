# S3 清理前基线与验收

日期：2026-10-09。状态：**COMPLETE_WITH_USER_ACCEPTED_EXCLUSIONS / S4_PENDING_USER_DECISION**。

[总体计划](../MASTER_PLAN.md) · [S3 路线](../03_BACKUP_BASELINE.md) · [机器验收](S3_verification.json) · [S4 交接](S4_HANDOFF.md)

## 备份与源码稳定性

快照：`D:\code11\IRI01\IRI03\INR1-1-run65_backup_v193_20261009_S3`。

- 复制 28,464 个文件，共 1,708,339,512 字节；逐文件 SHA-256 核对通过。
- 30,202 个大型数据/产物文件只登记位置、大小与哈希，原位保留。
- 300 个不可读历史 pytest/临时路径按 D-S3-001 原位保护，不改权限、不删除、不迁出；备份覆盖范围不包含这些路径。
- 87 个元数据、凭据或运行缓存目录按清单排除；未发现目录联接/符号链接。Git 另有 bundle、状态、差异和文件清单，环境另有 pip freeze。
- 4 个核心源码从备份恢复到独立抽检目录，哈希一致；此结论不等于全部数据恢复演练。
- S2 的 160 个源码文件最终哈希全部一致，14 项既有 Git 修改状态保持一致；正式主/S6/身份清单权重哈希核对通过。
- 进程查询未取得 Python 命令行，不能据此断言没有并行训练；文件复制期间检查大小/mtime，最终复核源码和权重哈希。

完整映射见 [backup_manifest.json](backup_manifest.json)。外部绘图及历史分析代码映射在该文件的 source_roots 字段。

## 固定基线

环境：CPU、2 线程、seed 42、确定性算法；PyTorch 2.13.0+cpu、NumPy 2.5.3。所有产物位于 [s3_baseline_20261009](s3_baseline_20261009)，未写正式训练目录。

12 个测试文件共 **79 passed / 242 warnings**。覆盖恢复调度、M2W 域、Gram 校准、QC、profile split、密度单位、GIRO、ISR 和 static ETKF。警告包括 NumPy/Matplotlib 弃用及 GIRO 样例中的 All-NaN；详见 [日志](s3_baseline_20261009/regression.log) 与 [XML](s3_baseline_20261009/regression.xml)。

| 真实权重 | 固定样例 data loss | raw Gram loss | 结果 |
|---|---:|---:|---|
| main_A1S_N8D64 | 0.007932417094707489 | 0.6206293702125549 | PASS |
| FNDA_V8_N4D64 | 0.007295453920960426 | 0.3563276529312134 | PASS |
| FNDA_V8_N8D32 | 0.01073266752064228 | 0.5903464555740356 | PASS |
| FNDA_V8_N8D64_REPRODUCTION | 0.007932417094707489 | 0.6206293702125549 | PASS |

四组均严格加载，状态键/形状/dtype、输出和梯度已留存；无观测时精确回退、增量为零；M10/M01/M11 Gram coverage 均为 1。主模型与 N8D64 重跑 NPZ 的全部数组逐值一致。主模型走生产加载器；S6 按各自 manifest 架构构造并严格加载，避免生产加载器固定 N8D64 的假设。

模型基线梯度目标为 data loss + raw Gram，只用于清理前后数值对照，不代表生产总损失。120/199.9 km 固定查询覆盖前向；低高度 anchor 契约另由 test_m2w_domain 覆盖。旧 run66 专属固定权重测试未作为当前模型证明。

## 单步 Analysis 与状态恢复

[结果](s3_baseline_20261009/analysis_step_v4/result.json) · [日志](s3_baseline_20261009/analysis_step_v4.log) · [脚本](s3_analysis_step.py)

使用真实主权重、生产 stage/mode/optimizer helper，在固定合成观测上执行一次 AdamW 更新；诊断目标为 data loss + 正式 resolved_training 中的 Gram 权重 × Gram，权重为 1.5550329423250582e-06，更新前 loss 为 0.00793338194489479。

4 个参数改变，50 个冻结参数未变；损失、梯度和更新后状态有限。新建模型严格恢复后，模型、AdamW、CosineAnnealingLR 状态逐值一致；Python/NumPy/Torch RNG 恢复后的下一次抽样一致。正式权重未改。

前三次尝试的日志保留：第一次推理配置缺 resolved_gram_weight；第二次误从 manifest.config 读取，实际字段在 resolved_training；第三次模型包含非叶张量，deepcopy 不支持。最终使用重新构建模型并严格加载进行恢复。均为独立核查脚本问题，未修改生产源码。

这项检查不覆盖完整真实数据训练、生产恢复入口或全部生产损失；本阶段未重算论文全量 ISR/GIRO 指标，未独立运行模块自测。S4 应根据每批实际改动补充对应检查。

## S4 对照标准

同一 CPU/软件/种子/固定输入的模型输出、模式损失和保留梯度优先逐值一致；键、形状、dtype、掩码、冻结状态及 RNG 状态精确一致。出现差异先保留失败并定位，不根据清理后结果放宽阈值。方向损失链调整需另外捕获其受影响 helper 的完整返回、梯度与 RNG 对照；现有推理基线不能替代该项检查。
