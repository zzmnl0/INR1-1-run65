# Run66 Query-Local源头语义与观测表示修复计划

状态：实施中。本文档独立于旧goal plan；后续永久排除全域latent、跨query统一分析状态、固定全域分析单元及跨query状态融合。

## 固定语义与边界

每个query独立构造七维局地物理系数状态 $a_q\in\mathbb{R}^7$、局地物理模态 $M_q(x)$、观测矩阵 $Y_q$，并独立执行一次ETKF。冻结M2-O epoch 9的Background、D64/N8、Global R、top-8、窗口、precision floor、loss、QC成品及train-only profile清单。首轮不使用source-specific状态/decoder、自由MLP、FiLM、PCGrad、rank loss或非对角系数协方差；不重新QC，不增加forecast-transition。

M2-R/M2-S仅作为诊断证据。raw innovation定义为 $y-B(x)$，先于物理模态产生；本计划只能修复 $HX$、协方差与目标更新方向。raw innovation仍受限时归因观测代表性，不调ETKF。

## RSR-0：冻结与审计

保存代码、配置、checkpoint及样本manifest的SHA256。每个query必须拥有独立query中心、七维系数状态、$M_q/Y_q$和分析系数；禁止跨query共享或融合分析结果。

## RSR-1：实现正确性

1. Background端点token始终按`background_state_dim=64`写回；七维只属于ETKF系数状态。
2. query、FY、COSMIC和参考端点复用同一上下文构造路径；payload包含端点hmF2/NmF2、相对高度、SZA、LST以及Background一、二阶垂直导数。
3. Background导数用冻结Background在10 km步长上的确定性有限差分，边界使用单边差分。
4. 参考端点必须使用自身上下文；禁止静默回退query上下文。
5. legacy/M2-O不新增必填字段，旧checkpoint保持`strict=True`及逐元素重现。

停止门禁：标准D64 payload能进入七维query-local前向；生产与shadow对相同输入产生逐元素一致的raw modes、Cholesky和规范化modes。

## RSR-2：同样本Shadow

- V0：修正实现后的query-hmF2 Legendre固定字典。
- V1：每个端点使用自身$\eta=(h-hmF2(x))/190$，同时保留绝对高度、LST和SZA，仍使用四个Legendre垂直模态。
- V2：七个固定Background自适应模态：常数、归一化$-dB/dh$、归一化$\eta dB/dh$、低空紧支撑、纬向、周期经向、相对时间加SZA。

相同query、端点坐标和物理上下文下，FY/COSMIC模态必须逐元素相同；source/profile ID、邻域数及分层标签不得进入模态。V0/V1/V2的raw innovation必须逐元素不变。V1全门禁通过时优先V1；仅V1低空或夜间失败且V2全通过时选V2；二者均失败则停止。

## RSR-3：真实Query-Local门禁

废除profile-center代理。直接使用标准top-8 payload和生产query-local前向，对FY/COSMIC各512个train-only profile做profile级A/B双折；每个目标token是独立分析问题，profile为bootstrap独立单位，按来源×低/高空×昼/夜报告。

硬门禁：FY、COSMIC和联合白化$Y_q$有效秩配对改善的95% CI下界均大于0；第一模态能量不高于V0；主要分层有效秩恶化不超过5%；两折总体及样本充足分层RMSE比不超过1.01；相对rank-1的profile配对bootstrap 95%上界小于0；协方差符号准确率和留出方向率改善CI下界大于0；Gram误差不超过$10^{-5}$且raw Gram最小特征值大于$10^{-6}$；ETKF有限、正定、确定；零观测/零innovation/padding回退；顺序与source标签置换不改变结果。

判定：$HX$与方向改善而raw innovation不变，表示修复有效；仅$HX$改善则归因分析尺度或观测支持不足；$HX$不改善则物理字典失败。两单源未同时正确时不得归因联合竞争。

## RSR-4：训练授权

只有RSR-3全部通过，才训练唯一入选版本：先100批Analysis预检，再进行唯一一次五epoch development screen。train-only门禁前禁止读取development、locked-test或ISR。H4观测系统差异保持“无法确定”，不得由来源专属模态吸收。

## 依据

沿用现有Hunt、Janjić、Fan、Yang、Yue、Wang和Xu等引用；不新增网络文献。

## 实施记录（2026-08-03）

RSR-0/1已完成。M2-O epoch 9严格加载、全部张量有限且重复推理逐元素确定，SHA256为`533a1009c42a8867da230939884076d96041e58c313e8793145e80b6457612b6`。标准payload的D64/7维混写已修复；query/observation/reference使用同一端点上下文生成规则；参考端点缺失时新字典会显式失败，不再静默复用query上下文。旧legacy/M2-O配置默认值和参数结构不变。

RSR-2/3已按FY/COSMIC各512个train-only profile、profile奇偶双折、真实top-8逐query生产路径执行。报告中的历史字段`M2-R`仅表示当前物理shadow候选输出，不表示复用M2-R训练权重或profile-center状态。

- V0相对M2-O的观测有效秩预门禁通过，但方向、协方差符号、RMSE及rank-1门禁失败；因此V0不是候选。
- V1相对同样本V0的FY目标FY/COSMIC/联合有效秩均值差为`-0.0801/-0.1237/-0.0926`；COSMIC目标为`-0.1347/-0.1344/-0.1334`，六项95% CI均完全低于0。第一模态能量和主要分层门禁也相对V0失败。
- V2相对V0的对应有效秩均值差为FY目标`-0.1773/-0.1170/-0.1919`，COSMIC目标`-0.0566/-0.0987/-0.0932`，六项95% CI同样完全低于0；第一模态能量和主要分层门禁失败。

结论：V1/V2均未通过RSR-3，`selected_dictionary=null`、`training_authorized=false`。按冻结规则停止，不执行100批预检、五epoch训练、development、locked-test或ISR。当前证据说明“补齐端点hmF2/LST/SZA”与所列七模态Background字典不能修复表示问题，并且相对正确实现后的V0进一步降低观测有效秩；问题仍位于ETKF求逆前，但不能归因联合竞争或ETKF/R/N8。

正式证据：`isr_validation_outputs/run66-query-local-semantics-repair/rsr3_v0_report.json`、`rsr3_v1_report.json`、`rsr3_v2_report.json`和`rsr3_summary.json`。

## 回退与清理记录（2026-08-03）

确认M2-O epoch 9为唯一有效模型。保留D64端点Background token与七维系数状态分离这一正确性修复；V0/V1/V2字典已从默认配置、正式训练和development入口移除。失败字典只能由独立审计脚本显式设置`allow_failed_query_local_shadow=true`构造，普通模型构造会立即拒绝。审计脚本与既有报告继续保留，不得将历史报告字段`M2-R`误读为有效生产模型。
