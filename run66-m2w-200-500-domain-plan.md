# M2-W 200–500 km模型域实施与判定计划

## 版本边界

M2-V以提交`d80d7bdfb50935e16113d302a51193e6844384fd`和远程恢复标签`run66-m2v-production-d80d7bd`冻结；M2-O恢复标签`run66-m2o-production-c31cb36`保持不变。M2-W在分支`codex/run66-m2w-200-500-domain`实施。M2-V继续采用v12和120–500 km语义，M2-W采用v13、`strict_200_500_domain_v1`及闭区间200–500 km，二者禁止交叉恢复或静默加载。

## 模型与数据契约

训练入口必须显式指定`--model-domain 200-500`。高度域写入架构签名、run manifest、训练状态和训练摘要；resume、eval、绘图、ISR及GIRO入口均从checkpoint同目录manifest恢复配置，并核验阶段、SHA256、v13、D64/N8、Global R、有限性和严格参数加载。Background-only产物不得进入Analysis评估。

FY和COSMIC在共享Dataset及NeighborhoodIndex入口按`200≤h≤500 km`过滤，不重新QC、不改变date-blocked划分或profile身份。每条profile继续保留8个稳定高度token；exact token表示检索1800 km×1.5 h正支持内的全部ragged token，不使用profile center或top-k。训练与外部推理token只允许train和development profile，locked-test profile不参与模型输入。

D64/N8、7个活动集合方向、Global R、4096 token分块、Gram约束及连续Gaspari–Cohn局地化保持不变。IRI代理不重训。M2-W最终development遍历全部域内点，训练期8点profile采样不得进入checkpoint选择或最终评估。

## Background候选与Analysis

使用相同seed和日期划分，从零训练5个Background epoch：

- `run66-m2w-200-500-gate-off`
- `run66-m2w-200-500-gate-on`

两者禁止M2-V或其他外部seed。gate-on仅复用M2-V现有连续可信度门，不实施已记录的300 km回退核心及300–350 km过渡方案。两候选均须满足FY和COSMIC各自`RMSE(M00)≤RMSE(Raw IRI)`；合格候选按两来源平均CCC最大、RMSE最小、Pearson R最大进行字典序选择，完全相同时选择gate-off。若均不合格，不启动Analysis。

胜出候选续训10个Analysis epoch。最佳checkpoint按FY/COSMIC完整development的M11平均CCC最大、平均RMSE最小、平均Pearson R最大选择，任何非有限指标均不能成为最佳模型。

## 可视化与外部评估

Analysis完成后生成全球切片、hmF2/NmF2图及200–500 km EDP廓线。ISR只报告200–300和300–500 km；M11和Raw IRI采用仅由ISR、M11和IRI定义的公共掩码，M00采用独立诊断掩码。GIRO在200–500 km内为M11、M00和Raw IRI分别执行独立粗峰与精细峰搜索，不复用M11峰位。

最终有效性只比较M11与Raw IRI，M00仅作诊断：

\[
\Delta CCC=CCC_{M11}-CCC_{IRI},\quad
\Delta RMSE=RMSE_{IRI}-RMSE_{M11},\quad
\Delta R=R_{M11}-R_{IRI}.
\]

采用2,000次、seed=42的配对bootstrap：卫星按profile、ISR按时刻廓线、GIRO按站点–时刻记录重采样。依CCC、RMSE、Pearson R顺序检查95%区间；第一个不跨零的指标决定通过或失败，三者均跨零则判为不确定。只有FY、COSMIC、Jicamarca、Poker Flat及GIRO主要结果均通过，才允许冻结M2-W。

## 验证与执行约束

回归检查覆盖200/500 km边界、域外target/token排除、profile身份、v12兼容、v13错误域及Background-only拒绝、完整development、字典序选择、ISR公共掩码、locked-test token排除、GIRO独立寻峰及bootstrap确定性和三种判定。随后执行相关pytest、模型自检、语法检查、`git diff --check`和单批smoke test。

仅提交源码、测试和本计划；排除checkpoint、日志、评估输出、`reports/`、PPTX和临时目录。长训练每30分钟单次检查，不持续追踪、不自动重启。
