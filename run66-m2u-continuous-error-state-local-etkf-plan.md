# FSIA-INR M2-U 共享局地响应 ETKF 实施计划（最终执行版）

## 1. 修改目标

M2-U 在保留 M2-O 的 IRI 背景、D64/N8 正交因子异常、共享端点 density basis、Global R、raw innovation、QC 与日期阻断语义的前提下，将观测影响从 query-local top-8 更新改为共享锚点响应。对模式 (m\in\{\mathrm{M10},\mathrm{M01},\mathrm{M11}\})，若无观测 query (q_u) 与有观测锚点 (x_o) 具有相同的完整误差状态 \(\eta\)，并位于同一锚点核心区，则要求

\[
\eta(q_u)=\eta(x_o),\quad q_u,x_o\in\mathcal C_{m,r}
\Rightarrow G_m(q_u,x_j)=G_m(x_o,x_j),
\quad \delta_m(q_u)=\delta_m(x_o).
\]

这里严格共享的是对同一 innovation 的 ETKF 增益响应及其分析增量，不要求两个位置的 M00 背景值或最终电子密度相同。锚点核心区采用严格共享，重叠区采用连续过渡，支持边界连续回退至 M00。

## 2. 数据与模式目录

FY 与 COSMIC-2 继续使用已经通过审计的 QC 成品，不重新 QC，不读取 locked-test、Jicamarca ISR、Poker Flat ISR 或正在运行的 M2-O 全量训练输出。每条有效 profile 最多保留八个高度 token，但 M2-U 不再按 query 选取固定 top-8 profile；在 1800 km、1.5 h 连续支持范围内建立全部正支持 token 目录。训练和 development 的目标 profile 在对应来源目录与观测池中同时排除。

M10、M01、M11 使用三个逻辑上独立的锚点目录。M10 锚点只来自 FY，且求解只使用 FY；M01 锚点只来自 COSMIC，且求解只使用 COSMIC；M11 锚点为 FY 与 COSMIC 的联合目录，并在每个锚点处只分解一次联合系统。由此，单源结果不依赖另一来源的观测值、位置或锚点数量。

## 3. 锚点 ETKF 与状态依赖局地化

在锚点 (r) 处，观测异常由共享端点 basis 与锚点集合异常给出。对来源 (s)，正定充分统计量为

\[
C_{r,s}=\sum_j \pi_{r,s,j}y_{r,s,j}y_{r,s,j}^{\mathsf T},
\qquad
b_{r,s}=\sum_j \pi_{r,s,j}y_{r,s,j}d_{s,j},
\]

其中 (d_{s,j}=y_{s,j}^{obs}-B(x_{s,j}))。精度由球面大圆距离平顶核、时间平顶核、误差状态核、representativeness 和 Global R 共同给出；不增加人工高度紧支撑核。M11 的系统为

\[
A_{r,11}=7I+C_{r,\mathrm{FY}}+C_{r,\mathrm{COSMIC}},
\]

并通过同一个 Cholesky 分解求解 FY 与 COSMIC 两个右端项。M10、M01 使用各自单源系统，不能由 M11 删除一项得到。

误差状态嵌入由背景低维状态和空间天气上下文构成。实现中以 D64 背景特征的前七个归一化方向和空间天气特征幅度构成连续 \\(\eta\\)，状态相似性使用方向—幅度二次距离；状态核设置非零 floor，避免观测稀疏时完全切断响应。

## 4. 核心共享与连续插值

锚点响应权重 (w_{m,r}\) 保存在集合空间，而不是保存某个 query 的标量增量。query 端首先由平顶覆盖核确定物理支持，再以冻结的状态距离分数进行 Sparsemax 分配：

\[
\lambda_{r}(q)=\kappa_{flat}(d_{gc}(q,x_r)/1800)\,
\kappa_{flat}(|t_q-t_r|/1.5),
\]

\[
\alpha(q)=\operatorname{Sparsemax}\left((\log\lambda_r(q)-
\tfrac12d_\eta^2(q,x_r))/\tau_m\right),
\qquad \beta_r(q)=\alpha_r(q)\lambda_r(q).
\]

最终背景权重为 (1-\sum_r\beta_r(q)\)，分析增量为

\[
\\delta_m(q)=\\sum_r\\beta_{m,r}(q)\\,\\delta_{m,r}^{anchor},
\\qquad
\\delta_{m,r}^{anchor}=y_{m,r}^{anchor\\mathsf T}w_{m,r}.
\]

因此最后一个锚点离开 1800 km 或 1.5 h 支持域时，(λ_r\to0)，增量连续退回 M00。该结构只承诺场值 (C^0) 连续；Sparsemax 活动集切换处不宣称一阶导数连续，也不把插值输出标记为严格 query 处后验方差。

## 5. 代码实施与检查顺序

首先固定 `shared_anchor_response` 架构签名、M2-U 温度及支持尺度，并将温度作为 checkpoint buffer 保存。随后使用全正支持 token 目录构造来源独立锚点，接入锚点充分统计量、精确 M10/M01/M11 Cholesky 求解、query 端 Sparsemax 和 M00 混合。保留 Gram whitening 与经验误差结构损失作为训练/诊断项，但不让其改变锚点核心共享的数学语义。

开发评估需输出锚点数量、活动锚点数量、核心覆盖率、重叠区增益/增量差异、M00 边界连续性、HX 有效秩与条件数、增量 RMS 和峰值内存。模型输出中的锚点混合结果明确标记为“局地 ETKF 分析插值”。

## 6. 验收与回退

必须通过以下结构检查：相同状态和相同核心区的 query 获得逐元素相同增益与增量；M10 对 COSMIC 增删不变；M01 对 FY 增删不变；M11 每个锚点只构造一次联合系统；无锚点严格等于 M00；边界退回连续；Sparsemax、背景权重和锚点权重非负且归一；分块与完整目录结果一致；目标 profile 不进入目录；checkpoint 恢复后重复推理确定。

M2-O 恢复点永久保留在远程标签 `run66-m2o-production-c31cb36`。若 M2-U 的精度、方向率、单源隔离或连续性不满足要求，只能从该标签建立新的工作空间回退，不覆盖或删除历史提交。
