# M2-V 新解决顺序1：连续物理可信度门实施计划

Jicamarca 结果表明，原始 QC-v2 Background 在 120–200 km 夜间和 200–300 km 夜间均出现退化。其首要风险是卫星与 ISR 在该条件下的残差方向不一致，因此本轮先限制 Background 订正的可信度，不改变 Global R、D64/N8、LETKF 或观测 QC，也不启动 Analysis。

## 推荐顺序

1. 固定连续物理可信度门，先阻止低层夜间 M00 进一步恶化。
2. 若门控结果只回退到 IRI、未降低原有 bias，则显式增强地方时/太阳天顶角条件表达；若卫星监督残差方向仍与 ISR 冲突，再实施 FY/COSMIC 条件性偏差订正。

本轮采用 120–200 km 夜间强抑制、200–300 km 夜间连续过渡的方案。若 200–300 km 夜间仍不通过，下一版再将平顶回退核心扩展至 300 km，并在 300–350 km 过渡；本轮不自动重训该下一版。

## 连续可信度门

令网络原始订正为 \(\Delta_{\rm raw}\)，实际 Background 为

\[
M00=IRI+g_{\rm bg}\Delta_{\rm raw},
\qquad g_{\rm bg}=1-w_h w_n w_m .
\]

使用 \(S(u)=u^2(3-2u)\)（先截断到 \([0,1]\)）定义

\[
w_h=1-S((h-200)/100),
\]

因此 120–200 km 为平顶核心，200–300 km 连续恢复，300 km 以上 \(w_h=0\)。地方时
\(LT=(t+\lambda/15)\bmod24\)，夜间权重为

\[
w_n=S\left(\frac{\cos(2\pi LT/24)+0.2}{0.2}\right),
\]

在 18:00–06:00 保持夜间核心，并在晨昏附近连续过渡。地磁权重为

\[
w_m=1-S\left(\frac{|\sin I|-0.25}{0.25}\right),
\]

使磁赤道支持域内执行门控，支持域外保持原始 Background。模型同时保留 `background_residual_raw`、`background_trust_gate` 和实际 `background_residual`，结构损失作用于门控后的订正；门控前订正的 RMS、最大值及 200–300 km 过渡区统计用于发现补偿性放大。

## 语义与产物隔离

门控默认关闭，历史 manifest 缺少门字段时解析为关闭，保证旧 M2-V 推理逐元素不变。显式 `--background-trust-gate` 才能启用新语义 `qc_v2_date_blocked_train_only_continuous_trust_gate_v1`，其门语义为 `fixed_altitude_localtime_dip_smoothstep_v1`，运行目录为 `run66-m2v-qcv2-background-trust-gate-v1`。门控新运行禁止外部 Background seed；门参数、架构签名、训练状态和摘要必须一致，否则拒绝续训。Background checkpoint 只能用于 Background 诊断或同一运行的后续续训，正式 Analysis/ISR 入口要求 `completed_stage=analysis`。

## 指标与硬门禁

FY、COSMIC development 均报告全局 RMSE、bias、profile RMSE、订正 RMS，以及 120–200、200–300、300–500 km × 昼/夜分层的 Raw IRI、M00、ΔRMSE、Δbias、门权重和原始/门控订正 RMS。两来源全局 RMSE 不得劣于 Raw IRI；分层卫星指标用于解释，不替代 ISR 硬门禁。

Jicamarca 继续使用既有 2024 年 9 月 train-day、\(dN_e/N_e\le0.5\) 筛选，不更换诊断数据。120–200 km 与 200–300 km 的昼/夜四单元均须具有至少 100 个有限点、至少 3 个有效日期，并满足

\[
RMSE(M00)\le RMSE(IRI),\qquad |bias(M00)|\le|bias(IRI)|.
\]

门控为零而 M00 不等于 IRI，或门控非零却声称处于平顶核心，均视为实现错误；Background硬门禁失败时保留报告；经用户明确授权，本轮可继续进行诊断性 Analysis，但不得将该运行标记为已冻结模型。

## 验证与执行

先通过门函数边界、单调性、有限性、门关闭历史等价性、无 seed 的 \(M00_0=IRI\)、外部 seed/语义不一致拒绝、Background-only 不能进入正式 Analysis 的轻量检查，再启动五个 Background epoch。训练完成后运行 Background 卫星分层比较和 Jicamarca 四单元诊断。若 Background 门禁通过，可进入常规 Analysis；若门禁失败，只有用户明确授权时才可继续诊断性 Analysis，并保留 Background 摘要和门禁报告。

全程使用既有 QC 成品，不读取 locked-test 或当前外部 M2-O 输出，不重新 QC。Git提交和远程同步按用户后续明确的交付计划执行。

## 本轮执行状态

当前 120–200 km 夜间门控严格回退 IRI；200–300 km 夜间门禁仍未通过。已记录但暂不实施下一步方案：将夜间回退核心扩展至 300 km，并在 300–350 km 连续恢复。本轮 Analysis 仅用于判断 M11 是否能在有观测区域纠正 M00，结果不能覆盖无观测区域的 Background 风险。
