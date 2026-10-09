# S2 依赖与处置清单

日期：2026-10-09。状态：**S2_COMPLETED / S3_PENDING**。

[总体计划](../MASTER_PLAN.md#阶段与关系) · [S2 路线](../02_DEPENDENCIES_SCOPE.md) · [逐文件清单 CSV](scope_inventory.csv) · [依赖与符号 JSON](scope_dependencies.json) · [决策记录](decisions.md) · [S3 建议](S3_HANDOFF.md)

## 基线与范围

当前工作区已由用户通过 N8D64 敏感性重跑确认：参数与结果一致，为训练源码。本阶段沿用这一结论，读取源码、配置、实验记录和图表来源，没有导入训练项目、加载权重或执行模型。S1 的历史快照/缓存哈希缺口不再用于否定训练源码身份。

本次静态清单覆盖根目录 Python 文件及 `inr_modules/`、`isr_evaluation/`、`traditional_static_etkf/` 内共 160 个 Python 文件，另记录根目录 PowerShell/文档命令引用、11 个代码副本目录和 S1 论文生成器。它不是整个仓库所有历史产物的删除清单。动态依赖及手动调用通过专项审查补充，未证明无用途的内容留在原位。

已按用户补充要求加入 [零权重与禁用分支处置](zero_config_review.md)：29项参数、主模型及3组S6有效配置矩阵，详见 [逐参数清单](zero_config_inventory.csv)。文件级KEEP不排除内部未生效专用计算链的精简。

自动分析从 7 类入口得到 39 个文件的保守导入并集；条件导入也计入，不能据此认定每个分支运行过。逐文件记录包括路径、职责、调用者、工作流、配置键、序列化核查位置、证据、处置及验证要求。函数和分支级处置见下表及三份专项报告。

## 当前工作流和依赖关系

| 身份 | 入口与主要关系 | 输入和输出 | 保留要求 |
| --- | --- | --- | --- |
| W1 训练／恢复 | `main_fsia.py` → `train_fsia.py` → FSIA 模型、sliding_dataset、FY/COSMIC、IRI、损失、训练合同 | 原 NPY/QC/index、OMNI、split → 模型、优化器状态、manifest/summary | 保留 Background→Analysis、精确模态损失、Gram 校准、恢复训练；实际配置按主/S6记录解析 |
| W2 推理／绘图 | `plot_fsia.py` → checkpoint loader、可视化及数据管理器 | 权重＋同目录记录 → 密度与图件 | 严格加载、坐标、物理单位及输出路径不变 |
| W3 GIRO | `evaluate_giro_peak.py` → checkpoint_io、sliding_dataset、peak_qa、evaluation_stats | 正式观测池、主权重 → cache、合同、统计 | 保留 common-valid 总体、shared-full30 和两类峰值有效性 |
| W4 ISR | `isr_evaluation/main_isr_eval.py` → loader/coord_convert、model_query、metrics、peak_qa、plots | Jicamarca/PokerFlat、权重 → cache、报告 | 保留几何、剖面上下文、无效值、bootstrap及契约 |
| W5 static ETKF／IRI | `traditional_static_etkf/{cli,core,runner}.py` | 静态总体、正式 FNDA 缓存、观测 → ETKF、IRI、指标 | 比较链整体保留；不混用 A1S2 或背景诊断缓存 |
| W6 数据处理 | `qc_profile_data.py` 与 data_managers | FY/COSMIC 原输入→QC、索引、报告；IRI/OMNI辅助 | 不把归档 cosmic_qc.npz 替换训练 NPY；保留列语义和profile ID |
| W7 模型自测 | `inr_modules/mdia/fsia_model.py` 的模块入口 | 小样例及实际要求的 IRI 依赖 | S3运行前检查副作用；不假定所有自测均无数据依赖 |
| W8 论文图表与消融 | 19 个去重生成器＋RO toggle/E1/E2/W3/S6 | S1 23 项图表来源及各实验预测 | 旧目录名称不影响保留；S6三个N/D配置与独立端点实验均保留 |

W1–W7 的具体导入行号和调用方见 JSON；W8 的绝对路径、源文件哈希见 S1 `model_identity.json.result_lineage`。S6 N4D64/N8D32 使用自身快照/控制器路径，不能强行替换为限定 N8D64 的主权重加载入口。

## 必须保留的语义与兼容关系

| 对象 | 证据／条件 | 处置 | 验证要求 |
| --- | --- | --- | --- |
| `legacy_feature_increment`、`endpoint_context_symmetric` | 主模型及 S6 配置实际使用；不是依据名字可删的旧分支 | 保留 | 固定输入的模式输出、状态键与形状一致 |
| `SpectralSWBranch`、`use_sw_freq=True` | 模型构造 `fsia_model.py:839`，主/S6启用 | 保留 | SW输入、频谱输出及模型前向一致 |
| exact mode loss、Gram loss、random_profile | 主/S6训练合同启用，`w_time_analysis=0.0033` | 保留 | 校准、有限梯度、损失及采样行为基线 |
| D/N 配置 | N4D64、N8D32、N8D64均用于论文 | 保留可配置结构 | 各自权重和加载路径分别验证 |
| `checkpoint_io.py`、`v14_contract.py` | 主权重格式14，strict加载及域/输入身份检查 | 保留 | 键名、shape、dtype、IRI冻结及manifest/summary校验 |
| `profile_split.py` | GIRO、ISR、ETKF shared-full30调用 | 保留 | 日期/剖面split及allowlist一致 |
| `date_split.py` 和日期划分资产 | 日期划分相关检查及训练合同 | 原位保留，实际模块调用继续按入口核实 | 20/5/5日期身份不变 |
| `inr_modules/evaluate_giro_peak.py` | 根入口的兼容转发 | 保留 | 调用方式保持兼容 |
| `background_m00_eval.py` | 比较缓存工具直接调用 `_metrics/_reporting_masks` | 保留共用部分 | 指标与mask一致；不能整文件迁出 |
| M00 observation calibration | main/train存在无条件导入，禁用配置不等于无依赖 | 保留模块，内部专用分支另审 | 导入、默认禁用路径及比较调用可用 |
| IRI峰值管理器 | 包括函数内延迟导入和动态torch获取 | 保留运行时功能 | 峰值输入、插值、单位一致 |
| Analysis vs resume checkpoint | Analysis为state_dict；resume另含optimizer/scheduler/scaler/RNG | 分开保留格式与加载机制 | 原分析权重严格加载；恢复状态完整性 |

详细证据：[训练审查](s2_training_review.md)、[数据与兼容审查](s2_compat_data_review.md)、[评估与绘图审查](s2_eval_workflows_review.md)。

## 分支级清理候选及关联修改

用户已确认：独立后续分支备份后迁出，共用部分保留。以下是 S4 的候选修改范围，不是现在执行迁出的命令。

| ID | 位置／符号 | 关系与建议 | 放行条件 |
| --- | --- | --- | --- |
| C01 | `main_fsia.py:609–624`；train `_a1s7_enabled`、`_resolve_a1s7_update_budget` | A1S7分支与adapter gate、预算、deferred合同、专用测试成组迁出 | S3完成，确认主/S6不启用；共用函数保留 |
| C02 | `main_fsia.py:586–608`；train `_a1s8_enabled`、anchor/exposure及epoch选择 | A1S8 mixed gate与loader/anchor/测试成组处理 | 不删除主路径仍调用的采样或背景选模实现 |
| C03 | `main_fsia.py:571–585`；train `_a1s9_enabled` 及校验 | A1S9复用A1S7 adapter及A1S8 anchor | 按共享依赖组处理，不能仅删除入口条件 |
| C04 | `run_a1s_natural_epoch.py`、`test_natural_epoch_sampler.py` | 独立natural-background流程成组迁出候选 | 保留当前主训练采样、混合数据及S6所需部分 |
| C05 | `run_a1s_m00_residual_correction.py`、`prepare_m00_residual_reliability.py`及测试/启动脚本 | 独立残差修正流程迁出候选 | 共用background_m00_eval等原位保留；记录产物位置 |
| C06 | `run_m00_condition_diagnostic.py`、`run_m00_low_budget.py`、`run_m00_luna_repair.py` | 独立M00实验入口迁出候选 | 保留合同、测试、失败日志；不影响论文比较工具 |
| C07 | `fsia_model.py:990–1127,1237–1375` 的condition adapters | 专用构造与forward分支可在对应实验迁出后简化 | state_dict和实际构造路径验证通过后才移除符号 |
| C08 | checkpoint格式12/13/15、A1S6转换和别名 | 当前未证明所有外部调用已退出 | 暂保留；不能将v14主格式一并删除 |
| C09 | `main_isr_eval.py:756,811–846` 的legacy MDIA加载 | 存在自定义config可达入口；候选收敛 | 核对调用者、测试和mdia_model，再决定，现阶段保留 |
| C10 | IRI自测样板、package/fallback双导入 | 不影响科学功能的简化候选 | 先确定直接执行和包执行的保留命令，不能破坏手动入口 |
| C11 | covariance moment／empirical covariance禁用损失链 | 主/S6均关闭，专用校准、计算、记录列为候选 | 保留共用分层/Gram依赖和必要checkpoint兼容读取 |
| C12 | direction loss禁用计算链 | 主/S6均关闭；Gram校准调用的 `_paired_analysis_losses` 仍计算并返回未使用的方向损失 | 同步简化共享返回契约与调用者；比较Gram校准结果、梯度和随机状态，不直接删函数 |
| C13 | 零混合比例、关闭gate及M00开关 | 与C01–C07后续研究族成组收敛 | 保留关闭状态下的恒等/回退和所需state_dict结构 |
| C14 | 有效权重为0但仍计算的路径 | 按运行配置及消费者逐处判定，见Z04 | 无共享/副作用后才跳过；固定损失、梯度、RNG及日志对照 |

CSV中10个Python文件标为 ARCHIVE_CANDIDATE；JSON另外列出3个PowerShell候选。没有在本阶段断言其中所有共享实现都可删除。执行到S4时需按上述组及调用者完成最小修改，不能逐行机械删除。

## 文件级分类的含义

- KEEP：核心导入关系或人工核实的必要依赖；内部可包含待简化分支。
- KEEP_TEST：保留工作流的既有回归检查。
- ARCHIVE_CANDIDATE：对应用户已批准退出日常维护的独立后续研究族，仍以S3备份和S4调用方核对为前提。
- RETAIN_PENDING_EVIDENCE：静态闭包外或用途未充分排除，默认原位保留；这不是让用户逐文件猜用途，也不阻止备份。今后若要迁出，必须补证据并更新清单。

清单覆盖全部160个Python文件；数量以 `S2_verification.json` 为准。未导入的历史审计、P0B工具和专用测试多数归入最后一类，历史失败记录全部保留。

11个 `inr_modules - 副本*` / `isr_evaluation - 副本*` 目录已逐Python文件对照当前代码；每个目录都有与当前不同的文件，不能按名称整批当作重复删除。详细哈希及匹配见JSON，暂保留并纳入S3来源登记。

## 跨目录和动态依赖

1. 19个论文生成器跨 `build/output`、V13/V15/V17 sources、`tmp` 及外部 restored 树。保留原路径关系；旧目录及临时目录名不作为清理理由。
2. 已发现 `figure2_v13.py → paper/revision_v2/build_coverage.py`、`restratify_v15.py → frozen_p1_statistics.py`、`fnda_v8_s5_tables.py → fnda_v8_complete_s4.verify_s4_contract` 等动态/相邻文件依赖。
3. 外部 nature-figure 绘图辅助脚本与 restored 源码属于外部依赖。S3登记路径、版本/哈希和可用性，不在原位置迁移或修改。
4. S6 experiment_manifest 绑定 `history/controller/fnda_v8_sensitivity_eceeb2324a02a4ab.py`。当前 `build/output/fnda_v8_sensitivity.py` 的 prepare 会复制/修补快照；不能通过随意执行prepare来“检查依赖”。
5. 尚未完成每个外部生成器的完整递归依赖闭包；S3备份必须按其所在完整源码目录保护，若做裁剪则先补全闭包。不能声称已实现最小独立发布包。

## 验收与下一阶段

160个文件已解析、记录导入/调用方/配置键及哈希；补充人工分支审查与外部工作流说明。两项本轮用户决策已记录。训练源码未修改、未执行模型或测试。本阶段完成的是清理范围审查，不是行为一致性验证。

建议进入S3，先固定完整备份及路径映射，再建立原权重、固定输入、数据索引、比较统计和恢复训练的对照。暂保留项一并保护，不在备份阶段抢先删除。是否进入S3由用户决定。
