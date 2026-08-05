# run66 改进方案：无泄漏、受控增益与连续残差

## 1. 结论与范围

run65-profile-fixed 已验证真实 profile 分组和 FY/COSMIC 非零分支，但尚未证明模型能可靠泛化。验证集 `RMSE=0.1549` 明显优于 IRI 的 `0.3800`，独立 ISR 改善却很小；Jicamarca 的 hmF2 MAE 由 `24.11 km` 退化为 `30.65 km`，夜间 120–300 km RMSE 由 `0.4717` 恶化为 `0.5401`。图像中的观测追随、廓线失形和时间断层与此一致。

run66 必须从头训练，不加载 run65 权重或优化器状态，不覆盖任何 run65 文件，不自动重做旧 P2/P3。首要目标是消除目标泄漏和错误增益反馈；网络扩容、全球共享分析状态和恢复已删除的 PeakHead/GIRO 训练均不在本轮范围。

## 2. run65 基线问题与 run66 决策

| 当前实现 | 问题 | run66 决策 |
|---|---|---|
| FY 使用 `random_split` 按点划分 | 同一 profile 跨训练/验证 | 改为按 `profile_id` 划分 |
| COSMIC 目标按日期划分，但邻域索引含全量数据 | two-pass 可选中目标自身 profile | 训练索引仅含训练 profile，并支持排除目标 ID |
| 邻域补零，但注意力无 token mask | 无效 token 参与注意力 | 显式传入 token 有效 mask |
| `R_FY_net/R_COSMIC_net` 已冻结，输出又被 `detach` | 只是固定、不可解释的增益校准，不是可学习 R | run66 活跃路径改用来源独立、可审计的有效 R |
| `K_FY/K_COSMIC` 是 `ens_var/(ens_var+R)` | 不代表 mask 和缩放后的真实更新 | 仅作诊断，新增实际来源更新量 |
| `h_analysis` 无观测时仍含 `delta_mean`，decoder 仍可输出残差 | M00 不是严格 IRI | 最终残差乘双源可用 mask，M00 硬回退 |
| 有效损失仅为观测、背景、IRI 隐特征重建和 peak alignment | 背景每 10 batch；peak alignment 前向未传邻域 | 背景每 batch；用观测条件 mini-column 约束替换 peak alignment |
| uncertainty 输出参与 gate 输入 | 密度路径与方差路径相互反馈 | 首轮关闭 uncertainty；后续启用时与 gate/R 解耦 |
| `main_fsia.py` 硬编码 run65-profile-fixed 和续训路径 | 会误续训或写入旧目录 | run66 使用显式新目录，默认 `resume=None` |

已删除的 `w_ne_vert_smooth`、`w_residual_smooth`、`w_peak_smooth`、`w_shape`、PeakHead、GIRO 训练加载器和 SALR 活跃路径不恢复。run66 架构已删除 `proj_pre`、`crf_alpha` 和恒为零输入的 `proj_frame_offset`，因此禁止加载 run65 权重。

## 3. P0：先修数据与验收语义

### 3.1 Profile 级隔离

1. FY 从 clean3 索引 6 读取 `profile_id`，COSMIC 沿用索引 5；ID 只作元数据，不进入模型特征。
2. seed 固定为 42，FY/COSMIC 分别按 profile 划分 train/validation；同一 profile 的全部高度点只能属于一个集合。
3. 训练和主验证使用仅由 train profile 构建的来源索引：
   - train target 查询同源邻域时排除自身 `profile_id`；
   - validation target 不存在于 train-only 索引，无需借用其他 validation profile；
   - ISR/P2 推理仍可使用当时全部合格 FY/COSMIC profile。
4. 保存 split manifest：数据 SHA256、profile 数、train/validation ID 摘要及 seed。旧邻域预计算缓存全部失效。
5. 保持已确认语义：对称 `±1.5 h`、局部纬经窗口、无邻域允许退化为无观测，不建立全球共享状态。

### 3.2 邻域接口

`FYNeighborhoodIndex` 与 `COSMICNeighborhoodIndex` 统一支持：

- `exclude_profile_ids`；
- token 有效 mask；
- profile/token 距离权重；
- 仅供测试和诊断的 selected profile IDs。

预计算与在线查询必须走同一 featurize 函数并给出逐元素一致结果。ISR、训练验证以及后续 P2/P3 继续复用这两个正式索引类，不另建旁路索引。

## 4. P0：修正同化增益和 IRI 回退

1. 在 FY/COSMIC mask 与来源缩放后分别计算 `update_FY`、`update_COSMIC`，记录其范数、方向和覆盖率；不再把 `K_*` 当作实际影响。
2. 定义 `obs_available = has_obs OR has_obs_cosmic`，最终 `Ne_delta` 必须乘该 mask。无观测时 Analysis 与已冻结 Background 逐元素一致。
3. 删除训练损失对 `K_FY` 的 `trust_iri` 反馈。IRI锚定独立于ETKF增益，且不使用逐点高度权重。
4. gate 不再读取 `log_var` 或自身 `Ne_delta_raw`；这些量可监控，但不能反向决定密度更新强度。
5. source dropout 只生成 M10/M01/M11，训练单源稳健性；M00 已由硬不变量保证，不浪费训练 batch。

## 5. P1：固定有效 R

不复活随机或可学习 R 网络。FY/COSMIC分别使用一个正的固定有效观测方差；Background训练结束后，按训练profile的 `Background-observation` 残差MAD稳健校准，限制标准差在0.05–0.40 dex，并写入checkpoint buffer。Analysis阶段不得联合学习R，避免R塌缩。

FY沿用已经完成QC的 `clean1` 物理值；`clean3`仅提供profile ID。COSMIC目前只执行有限值、正值和profile ID合法性检查，尚未完成专项科学QC，因此R只能吸收总体残差尺度，不能替代profile级质量控制。

## 6. P1：两阶段损失

FY/COSMIC均使用profile内平均、profile间等权的 `Huber(delta=0.2)`，两来源各占观测损失的50%。

- Stage 1 Background：`L_B=L_obs+0.02L_IRI+0.02L_vert(r_b)+0.01L_time(r_b)`，其中 `r_b=Background-IRI`；背景残差由前向硬限制在±0.5 dex。
- Stage 2 Analysis：冻结Background，`L_A=L_obs+0.01L_inc+0.05L_vert(d)+0.02L_time(d)`，其中 `d=Analysis-Background`；分析增量默认硬限制在±0.3 dex。
- 每个batch抽取32个位置，以20 km和1 h固定间隔构造垂直、时间triplet，仅惩罚残差二阶差分。
- source dropout按profile采样M10/M01/M11=`1:1:2`；M00由硬mask保证。
- 主模型关闭heteroscedastic NLL。可选不确定性只能在冻结密度模型后单独训练。

## 7. 分级实验

所有实验固定 seed、profile split、QC 阈值、数据身份和 ISR 时间范围；先在固定 10% train profile 子集上筛选。

| 实验 | 唯一新增改动 | 通过条件 |
|---|---|---|
| R66-A | profile split、自profile排除、严格M00、两阶段训练 | 无泄漏且四模式语义正确 |
| R66-B | A + 固定R校准、profile级source dropout、gate解耦 | 观测影响非零但受控 |
| R66-C | B + 垂直mini-column、时间triplet | 时间断层和廓线形变下降 |
| R66-D | C + 增量上限0.2/0.3/0.5 dex消融 | profile RMSE和独立ISR均通过 |
| R66-U | D + 解耦 uncertainty（可选） | 校准改善且密度结果不退化 |

每级依次执行：单 batch 前向/反向、关键参数梯度审计、500–2000 step 小样本过拟合、短训。只有 R66-D 通过后才启动全量 run66。

## 8. 验收门槛

- 数据：FY=`75,766`、COSMIC=`105,182` 个真实 profile；train/validation ID 交集为空；目标 ID 不出现在同源邻域。
- 验证：按 profile 等权报告 FY/COSMIC MAE、RMSE、CCC 和相对 IRI skill；最佳 checkpoint 不再由逐点随机 MSE 决定。
- 四模式：M00 与 IRI 的最大绝对误差不超过数值容差；覆盖点 M10/M01 均有 `>1e-6` 且有限的影响；M11 不恒等于任一单源。
- 梯度：FY/COSMIC encoder、`H_FY/H_COSMIC`及ETKF扰动参数均有有限非零梯度；固定R不得有梯度。
- 独立 ISR：逐点 RMSE/CCC 不差于 run65；Jicamarca hmF2 MAE 不差于 IRI 的 `24.11 km`，夜间 120–300 km RMSE 不差于 `0.4717`；PokerFlat 分日夜和高度报告。
- 连续性：同站分析残差二阶差分 MAD 比 run65 降低至少 30%，低频 RMSE 恶化不超过 1%。
- 工程：新目录为 `checkpoints_fsia/run66-etkf-loss`；严格加载、参数有限、重复冻结推理确定；保存配置、数据/代码身份、训练日志、最佳epoch和checkpoint SHA256。

ISR 只作最终独立门槛，不参与 checkpoint 选择或反复调参。R66-D 未通过前，不重做 P2/P3，不修改学生模型，不进入 P4。

## 9. 风险与控制

- 严格隔离后验证指标大幅变差是预期现象，不能退回有泄漏划分。
- 自 profile 排除和 train-only 索引会降低覆盖率；无局部邻域应回退 IRI，而不是放宽为全球共享。
- QC 可能误伤真实不规则体；必须按质量分层报告保留率和 ISR 结果，不以“更平滑”单独判优。
- mini-column/triplet每batch增加6×32个查询点及邻域检索，是主要训练时长风险。
- run66 从头训练且架构不兼容run65；历史checkpoint只作证据，不加载、不覆盖。

## 10. 2026-07-24 实施状态

代码已完成上述两阶段损失、profile等权采样/验证、固定R校准、profile级source dropout、硬残差上限、严格M00、ETKF唯一观测路径及断点阶段恢复。首个Background和Analysis epoch的前100个batch写入 `batch_diagnostics.jsonl`，包含原始/加权损失及解码器梯度范数；辅助/观测梯度比超过30%时报警。最佳checkpoint按FY/COSMIC profile RMSE均值选择。

已加入profile复制不变性、线性/尖峰二阶差分、M00和双源单路径梯度回归。尚未执行全量训练、0.2/0.3/0.5 dex消融、COSMIC专项QC、P2/P3或独立ISR验收。

## 参考依据

- Yang et al. (2009)：COSMIC profile 的 MD、顶侧梯度与异常 profile 质量控制。
- Yue et al. (2010)：Abel 反演在低纬、低高度及 EIA 附近存在结构化误差。
- Fu et al. (2023)：hmF2、峰数、MD、噪声因子和顶侧梯度联合 QC。
- Shi et al. (2022)：多源观测融合前的系统差异校准。
- 本地资料：`D:\0文件\0文献\GNSS11\同化_重构`、`D:\0文件\0文献\GNSS_QC`。
