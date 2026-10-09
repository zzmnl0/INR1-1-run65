# S2 评估、绘图、比较与敏感性工作流审查

所属阶段：[S2](../02_DEPENDENCIES_SCOPE.md)；服务于总清单，不是删除授权。2026-10-09，仅静态读取；未运行模型、评估、绘图或历史控制器。用户本轮确认当前工作区在敏感性实验中重跑 N8D64、参数与结果一致，作为训练源码接受；下列历史控制器与快照仍应保留各自身份。

## 保留工作流与依赖

|入口/类别|调用与配置条件|输入、输出及跨目录关系|处置与后续验证|
|---|---|---|---|
|`plot_fsia.py`|`load_fsia_analysis_checkpoint`、FY/COSMIC 邻域索引、SpaceWeatherManager、可选 IRIPeakManager、`visualization_mdia`；ISR overlay 为可选开关|必须显式 `--checkpoint`；默认输出该权重目录下 `plots`；全球切片、峰值图和廓线；可选读取外部 Jicamarca 原始数据|保留；测试输出路径定向到临时目录、密度单位、固定查询结果；头部旧 run66 注释可更新，不能据此删实现|
|`evaluate_giro_peak.py`|checkpoint_io、sliding_dataset、evaluation_stats、`isr_evaluation.peak_qa`、FY/COSMIC、IRI 与空间天气|显式权重及 GIRO 源数据；共享 full30 观测池、基线权重及哈希、resume、postprocess-only；输出 `giro_peak_cache.npz`、报告、契约|保留全部正式峰值 QA、共同总体、哈希和恢复语义；关联 `test_giro_shared_full30.py`、`test_density_plot_units.py`|
|`inr_modules/evaluate_giro_peak.py`|从根 `evaluate_giro_peak` 导入入口|兼容模块式/历史命令调用；不是独立旧算法|保留兼容入口，只有调用清单证明没有使用后才考虑收敛|
|`isr_evaluation/main_isr_eval.py`|main→严格权重加载→ISR loader/coord_convert→model_query→metrics/peak_qa→plots；bootstrap 来自 evaluation_stats|权重、Jicamarca/PokerFlat 数据；checkpoint 或 shared-full30 观测池；`--shared-full30-profile-split` 为旧日期 checkpoint 的显式覆盖；输出缓存与报告|保留 FSIA 路线及显式池覆盖；`--preflight-only` 可用于 S3 身份/加载检查，不执行本轮；关联 `test_isr_evaluation_qc.py`、`test_isr_loader_geometry.py`、`test_isr_model_query.py`|
|`traditional_static_etkf/{cli,core,runner}.py`|cli.prepare 生成静态集合；runner.run_isr 调用 core 分析与 ISR loader；profile_split、evaluation_stats、peak_qa 被共享使用|manifest、静态集合、canonical FNDA 缓存、ISR 原始数据；生成 `isr_etkf_cache.npz`、峰值缓存、指标及成本记录|整体保留，论文比较链必需；不得把独立 ETKF 与神经 FNDA 内部同化分支混为一类；保留 `test_core.py`、`test_runner.py`|
|`isr_evaluation/background_m00_eval.py`|直接构建 FSIA/IRI；比较脚本 `compare_a1s2_a1s_caches.py` 导入其 `_metrics/_reporting_masks`|外部/cache 数据；M00 背景和 static ETKF 参照；允许 diagnostic-only、未选中背景需显式允许|保留共享统计和对照能力；候选诊断功能归档前先解除导入依赖，不能整文件直接移走|
|`qc_profile_data.py`|S1 图表溯源生成器之一|原始 FY/COSMIC 质量控制及报告/中间产物|保留，非无用数据脚本；关联 `test_qc_profiles.py`|

具体手动命令骨架（仅记录，未执行）：`python plot_fsia.py --checkpoint <S1权重> --save-dir <临时目录>`；`python evaluate_giro_peak.py --checkpoint <S1权重> --save-dir <临时目录>`；`python -m isr_evaluation.main_isr_eval --checkpoint <S1权重> --preflight-only`；`python -m traditional_static_etkf.cli prepare --manifest <明确manifest> --output-dir <临时目录>`；`python -m traditional_static_etkf.runner isr --manifest <明确manifest> --static-dir <静态集合> --canonical-dir <正式FNDA缓存目录> --output-dir <临时目录>`。不得用当前默认配置替代 S1 固定配置。

## 历史分支候选（均未判定为可直接删除）

- `main_isr_eval.py:811` 的加载函数默认 `model_type='mdia'`，而顶层 CONFIG 为 `fsia`、另一检查路径默认也为 `fsia`。`else` 仍引用 `mdia_model.MDIA_INR_Model` 和 `_load_state_compat`（约 756、827–846 行）。这是旧 MDIA 支持的候选收敛点；外部调用可以传入自定义配置，必须核对调用者后成组移出，不能仅删除 mdia_model 文件。当前 FSIA 使用严格 checkpoint loader，应保持。
- `diagnose_m2v_low_night.py`、`diagnose_jicamarca_modes.py` 为专项诊断入口，导入 main_isr_eval 内部函数；`audit_m2w2_error_chain.py` 与 `summarize_m2w2_error_chain.py` 属于 P0B 历史审计，依赖 p0b_audit/counterfactuals 及合同。建议归为历史诊断候选，连同契约、测试和报告整体归档；不得丢失失败记录，且须核对当前测试调用。
- `prepare_spatial_dev5_isr_manifest.py`、`compare_a1s2_a1s_caches.py` 是特定方案/结果核查工具。后者存在对 background_m00_eval 的直接依赖。暂列待确认用途；用户未要求删除所有非主训练入口。
- `__pycache__/*.pyc` 为可再生产物；S3 快照记录后可清理，无模型身份价值。测试命名含旧版本不能作为删除理由。

## 图表生成器不能按目录名称批量移除

S1 `model_identity.json.result_lineage` 中有 **19 个去重生成器路径**，跨 `build/output`、v13/v15 sources、v17.2c sources、`tmp/v7_s3_figures`、外部 restored 树及归档检查代码。它们仍被 23 项正文/SI 图表的溯源记录引用。建议作为论文复现链保护对象，S4 若整合只迁移必要闭包并更新路径，不清空这些父目录。

发现的间接依赖：

- `figure2_v13.py` 动态加载 `paper/revision_v2/build_coverage.py`；仅保留 result_lineage 直接列出的文件不足以复现。
- `restratify_v15.py:53` 动态加载 v13 `sources/frozen_p1_statistics.py`，并断言 bootstrap `REPLICATES=20000`、`SEED=42`、指标集合。必须保留这些统计约束。
- `build/output/fnda_v8_s5_tables.py:14` 导入 `fnda_v8_complete_s4.verify_s4_contract`；不能把相邻 controller 都认作历史无用文件。
- 多个 figure 生成器硬编码 `C:/Users/12238/.codex/skills/nature-figure/scripts`，依赖外部绘图样式工具；S3 记录实际模块及哈希，S4 需要移植时再判断是否纳入项目。S2 未调用这些工具。
- 生成器使用 `Path(__file__).parents[...]` 和固定旧 run 路径；移动位置会改变输入/输出。未迁移前保留原位置，避免重跑覆盖正式图件。

## restored 树与 Table S6 的特殊身份

`D:/code11/IRI01/IRI03/INR1-1-run65-A1S-restored` 为外部证据依赖，不在当前清理目录内。Figure 7/S5 依赖 E2 NPZ，S4 依赖 E1 CSV 与 `support_diagnostic.py`；机制图生成器位于 supplement_v3/P3_mechanism。

Table S6 的 N4D64、N8D32、N8D64_reproduction 运行资料位于 `.../build/a1s_submission_v8/run_20260914_021617/experiments/sensitivity`，包括 runs、source_snapshot、inputs、共同评价总体、prediction.npy、bootstrap 和历史控制器。`workflow_state.json` 已记录 prepare passed/training completed/evaluation completed/summary completed（文件记录，非本轮复算）。

该目录 `experiment_manifest.json` 记录 driver 为历史冻结控制器 `history/controller/fnda_v8_sensitivity_eceeb2324a02a4ab.py`，不能与当前 `build/output/fnda_v8_sensitivity.py` 自动等同。后者含 preflight/prepare/run-training/run-evaluation/summarize，并且 prepare 会复制和修补 source_snapshot，训练通过 subprocess 执行。清理时将其作为受保护的实验编排入口及历史证据；不直接运行 prepare 验证是否“还能用”。

source_snapshot_manifest 说明快照来自 restored 树；用户已确认当前工作区 N8D64 复现。两项可同时记录：当前源码身份已获确认，敏感性历史快照/控制器仍以各自 manifest 为准。没有理由用快照覆盖当前树。

## S3 建议与验证缺口

S3 备份应覆盖当前代码及所有受保护生成器的间接依赖、运行配置与测试；外部 restored、图表缓存、权重和原始观测先按精确路径/哈希引用保留，不搬动或删除。针对 `main_isr_eval` 的 legacy MDIA 分支、诊断入口与绘图外部脚本形成逐项处置清单；尚未覆盖全部外部手动调用者的项目保持待确认。

本轮没有执行任何测试或前向推理；未递归扫描大型输出缓存、未核验所有外部脚本的递归依赖闭包。静态审查已足以确认：论文生成器、比较方法、历史审计不能按“旧目录/旧版本名”批量删除。
