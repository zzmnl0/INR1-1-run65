# S2 训练链路只读审查

身份：S2 训练与模型分支审查，供总体依赖清单合并使用。2026-10-09。

本报告接受用户确认：当前工作区在参数敏感性实验中重新运行 N8D64，参数和结果一致，因此作为已复现的训练源码维护。保留原主 run 与敏感性 run 的不同运行身份；不把主 run 初始 manifest 的 Background-only 值当作最终 Analysis 配置。

## 核查依据和当前合同

- `checkpoints_fsia/run67-m2w-120-500-analysis-a1s-10epoch/training_summary.json`：最终 architecture、training_protocol 和损失开关。
- `FNDA_V19_archive_scheme2/data/B_model_assets/weights/sensitivity/{N4D64,N8D32}/run_manifest.json`、对应 summary。
- `FNDA_V19_archive_scheme2/data/D_results/sensitivity_configs.json`：N4D64、N8D32、N8D64_reproduction 三种配置及已有权重加载结果记录。本阶段只读取历史检查结果，不称为重新执行通过。

主模型 D=64、N=8；敏感性还需 D=64/N=4 与 D=32/N=8。共同保留 `endpoint_context_symmetric` 密度基、`legacy_feature_increment` 状态语义、`orthogonal_factor` 扰动、`random_profile` 来源调度、`analysis_exact_mode_loss=True`、`w_time_analysis=0.0033`、`use_sw_freq=True`。主 summary 中 empirical covariance/direction loss 为 False，observation Gram loss 为 True。模型域 120–500 km、观测及峰值域 200–500 km，物理定位 1800 km/1.5 h，观测块 4096。训练 seed=42、Background 5 epochs、Analysis 10 epochs、Background 选择 `bias_constrained_v1`。

## 必须保留

| 部位 | 证据／调用关系 | 保留理由 |
|---|---|---|
| `main_fsia.py:258` 主入口 → `train_fsia.py:7336` | main 顶部直接导入 train_fsia | 训练、恢复和评估的入口；CLI→配置→manifest 关系必须整体维持 |
| `fsia_model.py:788` FSIA_INR_Model，`:245` NeuralETKFLayer | train_fsia 与 checkpoint loader 构建模型 | 参数名、tensor shape、严格加载兼容性 |
| `fsia_model.py:768` SpectralSWBranch | 模型构造器 `:839` 由 use_sw_freq 控制 | 正式配置 True，非无用频谱模块 |
| `fsia_model.py:59,94` solve_density_mode(s) | train_fsia 顶部导入 | 精确来源模式损失和局地同化 |
| `sliding_dataset.py:323` SlidingWindowBatchProcessor 与来源掩码辅助函数 | train_fsia `:32–39` 导入 | 局部观测、掩码、空邻居及物理定位 |
| `physics_losses_mdia.py:7,27` profile_huber_loss、second_difference_loss | train_fsia `:31` 导入 | 按 profile 聚合及平滑损失，不是未使用旧模块 |
| 日期拆分、Background bias selection、低空 IRI 先验 | 主 summary 的 training_protocol、date_split、selection | 保留论文训练合同和避免数据泄漏 |
| D、N 可配置逻辑 `fsia_model.py:797–810,853` | 三种 S6 配置 | 不硬编码仅 N8D64；保留 N4D64/N8D32 |

特别注意：`legacy_feature_increment` 名称中含 legacy，但它是正式 checkpoint 的实际语义，不可据名清理。`fsia_model.py:806–812` 的 shared_latent 固定 D64/N8 校验是条件分支，不能把其约束扩大到 S6 的其他配置。

## 后续分支清理候选

以下均是“归档／移出当前维护范围候选”，不是本阶段已批准删除项。

| 分支 | 位置及触发条件 | 连带范围与处理建议 |
|---|---|---|
| A1S7 双磁纬 adapter | main `:609–624`；train `_a1s7_enabled:599`、`_resolve_a1s7_update_budget:658` | 不在已核对主/S6 配置中启用；连同 adapter gate、预算、deferred contract `:1018`、专用 `test_a1s7_m00_b.py` 共同归档候选 |
| A1S8 mixed gate／anchor | main `:586–608`；train `_a1s8_enabled:610`、预算 `:631`、deferred `:1042`、exposure `:1180` | 跨 loader、epoch、选模和诊断，不能只删主入口条件；主调用点 train `:8326,8337,8735,8768`。专用 `test_a1s8_m00_preserve.py` 同步处理 |
| A1S9 strict baseline | main `:571–585`；train `_a1s9_enabled:622` 与 `:7394` 校验 | 使用 A1S7 adapter scope 与 A1S8 anchor，固定 cap=.05、anchor target=.20、960 updates；必须按共同依赖处理 |
| condition adapters／M00 candidate | fsia_model `:990–1127` 构造及 `:1237–1375` 使用 | 主/S6 manifest 未声明启用；候选但需 checkpoint_io 与 ISR 构造器联合审查，删除前保护原 state_dict 行为 |
| M00 observation calibration | main `:20` 无条件 import；train `:42` 无条件 import、`:533` 解析；config `:309–333` | 即使配置 disabled，模块仍有 import 依赖，不得先移走文件；与主程序入口、相关审查／比较工具共同界定 |
| 经验协方差／direction loss | 主 summary 为 False | 禁用不等于无用；先核对所有需保留的消融、诊断与 S6 manifest。Gram loss 主配置为 True，必须保留 |
| 顶层相对／直接 import 双路线 | train `:26–77` | 存在 package 与 fallback 导入；只有证实直执行入口不保留后，才可简化，避免独立脚本失效 |

建议总体决策：后续 A1S7/8/9 及其他非论文研究链路统一保存到活动源码之外的可恢复备份；S4 按“主入口参数→默认配置→训练分支→模型组件→loader/评估→专用测试”成组处理。若仍需日常运行这些研究路线，则不能作为当前清理目标，应保留或另立项目。

## S3 应固定的行为基线

1. 三种 D/N 配置原权重严格加载；参数名、形状、参数数目一致。
2. 固定输入 M00/M10/M01/M11 与空观测输出一致；无观测精确回退到冻结背景。
3. 120–200 km 低空先验、200 km 边界、物理邻域正支持及缺失来源掩码。
4. 同一固定 batch 的损失各项、梯度目标和一次优化更新；不需要重跑全训练。
5. seed、日期/剖面分配、Background/Analysis 选择规则与 resume 状态恢复。
6. 删除候选 CLI 原行为及保留 CLI 配置解析；不将旧 run manifest 无声改写。

本阶段仅源代码／历史 JSON 静态审查，未运行训练、加载权重或重新验证历史指标。根目录宽范围 rg 遇到历史 tmp 与 pytest 输出目录拒绝访问，后续审查改用明确源码和归档路径；这些拒绝访问项不作为“无引用”的证据。
