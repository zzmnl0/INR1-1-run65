# FSIA-INR M2-U 连续误差状态局地ETKF实施计划（最终核查修正版）

本修正版以M2-U研究目标、M2-O当前生产代码、冻结历史诊断和预期development结果为共同约束。计划描述的是下一阶段拟实施架构，不表示M2-U代码、辅助统计或训练已经完成。

## 1. 研究目标与方案边界

M2-U拟在M2-O基础上构建连续、状态依赖的局地化算子，使背景误差状态相似的四维位置对同一观测innovation具有相似的ETKF增益响应，同时保留平滑、宽范围的物理距离约束，避免非物理的远距离传播。其目标不是使两个query获得完全相同的分析增量，而是使二者对相同观测扰动具有相近的响应规律：

\[
d_\eta(q_1,q_2)\ll 1
\quad\Longrightarrow\quad
\|G(q_1,x_j)-G(q_2,x_j)\|\ll 1,
\]

其中，\(\eta(q)=(e(q),a(q))\)由7维误差方向和因子幅度共同描述背景误差状态，\(G(q,x_j)=\partial\delta(q)/\partial d_j\)为观测\(j\)对query \(q\)的线性增益响应，\(d_j=y_j^{\mathrm{obs}}-B(x_j)\)为相对于M00背景场的raw innovation。实际分析增量仍由观测值、背景偏差、观测误差及局地权重共同决定：

\[
\delta(q)=\sum_jG(q,x_j)d_j.
\]

M2-U不建立单一全域latent，也不在不同query之间融合分析状态。每个query仍独立求解局地ETKF，但局地观测影响不再由矩形时空窗口和hard top-8决定，而由连续物理先验与低维背景误差状态共同确定。

本方案保留M2-O的M00背景场、D64特征状态、N8集合、Global R、`orthogonal_factor`异常参数化、`endpoint_context_symmetric` density basis，以及M00/M10/M01/M11的精确语义。FY和COSMIC-2的QC成品、日期阻断、来源内目标profile排除、每个profile最多8个高度token等数据语义保持不变。当前外部M2-O训练目录、locked-test及ISR独立验证数据均不进入M2-U开发。

## 2. 总体模型框架

M2-U的总体链条为：

\[
\boxed{\text{IRI订正背景M00}}
\rightarrow
\boxed{\text{连续物理候选池}}
\rightarrow
\boxed{\text{背景误差状态表示}}
\rightarrow
\boxed{\text{连续状态依赖局地化}}
\rightarrow
\boxed{\text{query-local ETKF}}
\rightarrow
\boxed{\text{连续四维分析场}}.
\]

M00继续承担全球背景均值订正：冻结IRI proxy提供IRI电子密度及隐藏特征，Kp/F10.7历史序列、IRI峰值上下文和空间时间坐标用于形成连续背景场\(B(q)\)。M2-U只改变观测候选构造、局地化权重及相应训练约束，不改变背景阶段和ETKF基本求解。

分析阶段首先在宽物理支持内检索所有可能具有影响的FY/COSMIC profile，再对实际高度token计算连续物理权重。D64集合异常投影到7个独立因子坐标后形成端点误差状态表示，物理核与状态核的乘积决定每个观测token的precision。随后，FY与COSMIC分别形成充分统计量，并按M10、M01和M11的定义完成单源或联合求解。

## 3. 连续物理候选池

### 3.1 球面空间—时间核

设query与观测端点的纬度和经度分别为\((\varphi_q,\lambda_q)\)与\((\varphi_j,\lambda_j)\)。球面中心角和大圆距离定义为：

\[
\Delta\sigma_j
=
\arccos\!\left[
\sin\varphi_q\sin\varphi_j
+\cos\varphi_q\cos\varphi_j
\cos(\lambda_j-\lambda_q)
\right],
\qquad
d_{\mathrm{gc},j}=R_E\Delta\sigma_j.
\]

为避免反余弦在极小距离处的数值敏感性，实际计算采用与该定义等价的haversine形式，并对其参数限制在\([0,1]\)。空间和时间无量纲距离为：

\[
r_{s,j}(q)=\frac{d_{\mathrm{gc},j}}{L_s},
\qquad
r_{t,j}(q)=\frac{|t_j-t_q|}{L_t}.
\]

其中，\(L_s\)和\(L_t\)为宽范围物理先验尺度，不再表示矩形内等权的人工分析单元。首版预注册\(L_t=1.5\ \mathrm{h}\)，保持M2-O时间支持；取\(L_s=1800\ \mathrm{km}\)，近似包络赤道处旧\(\pm5^\circ\)纬度和\(\pm15^\circ\)经度矩形的最远角点。该选择只将人工矩形改为球面连续支持，不依据development调参。首版不设置紧支撑高度核：高度相关由endpoint-context density basis与背景误差状态核学习，避免以人工\(L_v\)再次切断跨高度误差相关。采用端点值与一阶导数均连续的紧支撑函数：

\[
\kappa(u)=
\begin{cases}
1-3u^2+2u^3, & 0\le u<1,\\
0, & u\ge 1,
\end{cases}
\]

并定义：

\[
\lambda_{\mathrm{phys},j}(q)
=
\kappa\!\left(r_{s,j}(q)\right)
\kappa\!\left(r_{t,j}(q)\right).
\]

该核在支持边界处平滑趋于零，因此候选profile在粗筛集合中进入或离开时，只要实际token权重已为零，就不会在ETKF充分统计量中产生跳变。\(L_s\)与\(L_t\)作为显式配置和checkpoint身份保存，但首版固定为上述值，不在同一训练中自适应学习，也不依据development结果调整。

### 3.2 安全粗筛与实际token判定

候选profile不能仅依据profile代表点是否位于支持域内判定。对profile \(p\)预先计算球面中心\(c_p\)、覆盖所有有效高度token水平位置的球面包络半径\(r_p\)，以及时间中心\(t_p\)和半宽\(\tau_p\)。粗筛只执行不漏检的充分条件：

\[
d_{\mathrm{gc}}(q,c_p)\le L_s+r_p,
\qquad
|t_q-t_p|\le L_t+\tau_p,
\]

通过粗筛后，必须使用每个有效高度token自身的经纬度和时刻逐个计算\(\lambda_{\mathrm{phys},j}\)；只有至少一个实际token满足\(\lambda_{\mathrm{phys},j}>0\)，该profile才对query产生观测贡献。高度虽不进入物理紧支撑核，但进入\(\phi(x_j)\)、\(F(x_j)\)和完整误差状态\(\eta(x_j)=(e(x_j),a(x_j))\)，从而参与观测异常、误差状态距离和最终增益。

M2-U取消“非零权重候选中的hard top-8 profile删除”。每个入选profile仍最多提供8个高度token，这是对单条EDP垂直支持的统一采样，不是profile级局地化截断。候选数量变化只发生在零权重边界，ETKF使用全部正权重token累积充分统计量。

## 4. 背景误差状态表示

### 4.1 复用M2-O的7维独立因子坐标

首版不新增独立的状态编码网络。M2-O的D64/N8正交因子参数化可写为：

\[
X=U\,\operatorname{diag}(s_0)\,C,
\]

其中，\(U\in\mathbb{R}^{64\times7}\)为正交状态基，\(s_0\in\mathbb{R}^{7}\)为M2-O endpoint-context symmetric路径使用的固定因子尺度，\(C\in\mathbb{R}^{7\times8}\)为零均值Helmert集合系数，并满足\(CC^T=7I_7\)。共享density basis在端点\(x\)处给出\(\phi(x)\in\mathbb{R}^{64}\)，于是7维列向量形式的观测因子坐标为：

\[
F(x)=\operatorname{diag}(s_0)U^T\phi(x)\in\mathbb{R}^{7}.
\]

其与生产ETKF观测异常严格满足：

\[
HX(x)=F(x)^TC.
\]

因此，\(F(q)^TF(x_j)\)表征当前集合模型给出的背景误差交叉协方差结构。归一化误差状态嵌入定义为：

\[
e(x)=\frac{F(x)}{\max(\|F(x)\|_2,\varepsilon_F)}.
\]

只有\(\|F(x)\|_2\ge\varepsilon_F\)的端点被视为具有有效误差方向，此时\(\|e(x)\|_2=1\)。低范数端点从状态相似性、长度尺度和增益响应loss中排除，其\(\lambda_{\mathrm{state}}\)回退为1，仅保留物理核与representativeness，而不是用近零向量制造伪状态距离。

归一化方向\(e\)本身不足以定义完整误差状态，因为两个端点可能方向相同但集合方差幅度不同。另定义对数因子幅度：

\[
a(x)=\log\!\left(\|F(x)\|_2+\varepsilon_F\right).
\]

M2-U局地化状态由方向与幅度共同构成，记为\(\eta(x)=(e(x),a(x))\)。\(e(q)^Te(x_j)\)对应模型内部的误差相关方向，\(a\)描述集合误差尺度；二者都不是电子密度数值相似性。Kp/F10.7历史、地方时、太阳天顶角、IRI背景与峰值上下文通过M00和endpoint-context basis间接进入\(F\)，从而允许误差相关结构随电离层状态变化。

### 4.2 方向与幅度均敏感的状态距离

首版不新增可学习Mahalanobis矩阵。对两个有效误差状态，方向距离与幅度距离分别定义为：

\[
d_{e,j}^2(q)=\|e(q)-e(x_j)\|_2^2
=2\left[1-e(q)^Te(x_j)\right].
\]

\[
d_{a,j}^2(q)=\left[a(q)-a(x_j)\right]^2.
\]

方向定义保留相关符号：同向误差状态距离较小，反向误差状态距离较大。不能使用\([e(q)^Te(x_j)]^2\)或其他消除符号的相似度，否则正相关与负相关端点会被错误视为同一响应状态。幅度项防止仅因方向相同，就把集合误差尺度明显不同的位置视为具有相同原始增益响应。

## 5. 状态依赖局地化与ETKF更新

### 5.1 状态核

状态核采用带下限的连续高斯核：

\[
\lambda_{\mathrm{state},j}(q)
=
0.05
+0.95\exp\!\left[
-\frac{d_{e,j}^2(q)}{2\ell_e^2}
-\frac{d_{a,j}^2(q)}{2\ell_a^2}
\right].
\]

下限0.05避免早期误差状态不稳定时完全切断物理上合理的观测联系；任一端点误差方向无效时直接令\(\lambda_{\mathrm{state}}=1\)。状态核进入precision时对\(e\)、\(a\)、\(\ell_e\)和\(\ell_a\)使用stop-gradient；误差状态的学习仅由经验误差结构、Gram whitening和增益响应约束驱动，避免模型通过缩放或扭曲局地化权重直接降低分析loss。

观测token的最终precision为：

\[
\pi_{s,j}(q)
=
\frac{
m_{s,j}
\lambda_{\mathrm{phys},j}(q)
\lambda_{\mathrm{rep},s,j}(q)
\lambda_{\mathrm{state},j}(q)
}{R_s},
\qquad s\in\{\mathrm{FY},\mathrm{COSMIC}\},
\]

其中，\(m_{s,j}\)为有效mask，\(R_s\)为来源对应的Global R，\(\lambda_{\mathrm{rep}}\)使用M2-U重新生成的train-only统计文件，representativeness floor继续固定为0.25。

### 5.2 连续充分统计量

令\(Y_s(q)\in\mathbb{R}^{M_s\times8}\)为来源\(s\)的观测集合异常，\(d_s\in\mathbb{R}^{M_s}\)为raw innovation，\(\Pi_s(q)=\operatorname{diag}(\pi_{s,j}(q))\)。每个来源分别累积：

\[
C_s(q)=Y_s(q)^T\Pi_s(q)Y_s(q),
\qquad
b_s(q)=Y_s(q)^T\Pi_s(q)d_s.
\]

联合系统为：

\[
A_{11}(q)=7I+C_{\mathrm{FY}}(q)+C_{\mathrm{COSMIC}}(q).
\]

M11仍只构造一次联合系统，并分别求解：

\[
A_{11}w_{\mathrm{FY}}=b_{\mathrm{FY}},
\qquad
A_{11}w_{\mathrm{COSMIC}}=b_{\mathrm{COSMIC}},
\]

\[
M11(q)=B(q)+y_q^Tw_{\mathrm{FY}}+y_q^Tw_{\mathrm{COSMIC}},
\]

其中，集合异常采用列向量\(y_q=X^T\phi(q)\in\mathbb{R}^{8}\)。M10与M01分别使用\(A_{10}=7I+C_{\mathrm{FY}}\)和\(A_{01}=7I+C_{\mathrm{COSMIC}}\)独立求解，不能由M11删除某一来源贡献得到。无正权重观测时，\(C_s=b_s=0\)，分析严格退回M00背景场。

当\(F(q)\)、物理核、状态核及representativeness插值均连续，候选集合只在零权重边界改变，且\(A(q)\succeq7I\)时，\(A^{-1}(q)\)、\(w(q)\)和\(\delta(q)\)均随query连续变化。该结论保证消除人为候选边界，但不意味着真实电离层增量在所有位置相同。

### 5.3 增益响应

联合模式下，第\(j\)个观测token对query的增益响应为：

\[
G_{s,j}(q)
=
\pi_{s,j}(q)y_q^TA_{11}^{-1}(q)y_{s,j},
\]

其中\(y_{s,j}=X^T\phi(x_j)\in\mathbb{R}^{8}\)为该观测token的集合异常列向量。M10和M01使用各自的单源系统矩阵。该定义直接对应\(\partial\delta(q)/\partial d_{s,j}\)，因而“相似状态具有相似响应”应约束\(G\)，而不是约束包含不同innovation的最终增量\(\delta\)。

## 6. M2-U统一辅助统计

### 6.1 必须重新生成统计文件

M2-O现有经验协方差与representativeness统计由矩形窗口和top-8 profile样本分布生成。取消hard top-8并改用连续物理核后，邻域样本总体、profile配对频率和加权矩均发生变化，因此旧统计文件不能用于M2-U。

M2-U须从train-only日期和冻结Background重新生成独立统计文件，不覆盖M2-O历史产物。该文件同时服务于representativeness和经验协方差/相关性监督，至少包含stable标记、协方差、相关系数、target与observation的profile-level均值和方差、bootstrap置信区间、target/neighbor profile数量、ordered profile pair数量、有效日期数量，以及Background checkpoint、FY/COSMIC NPY、profile index、QC report和日期阻断manifest的SHA256。representativeness不再沿用旧max-norm \(\rho\)轴，而在M2-U物理邻近度轴上对新stable单元连续插值，并继续映射到\([0.25,1]\)。

离线统计使用的权重严格限定为：

\[
\omega_{k}^{\mathrm{stat}}
=m_k\lambda_{\mathrm{phys},k}.
\]

统计阶段不得包含尚未由统计结果生成的representativeness权重、状态核或\(R^{-1}\)。这消除了“统计决定权重、权重反过来决定统计”的循环依赖。

### 6.2 保持profile-blocked统计层级

定义M2-U物理邻近度：

\[
u_{\mathrm{phys},k}=1-\lambda_{\mathrm{phys},k}\in[0,1].
\]

设\(p\)为target profile，\(n\)为neighbor profile，\(c\)为来源对、target高度层、observation高度层、地方时类别和\(u_{\mathrm{phys}}\)层构成的统计单元，\(k\)为该profile pair内的token配对。物理邻近度层边界在运行统计前根据固定train-only加权分布确定并写入统计文件，训练和development只读取，不重新估计。背景残差定义为：

\[
r_k=y^{\mathrm{obs}}(q_k)-B(q_k),
\qquad
d_k=y^{\mathrm{obs}}(x_{j,k})-B(x_{j,k}).
\]

首先在每个target-neighbor profile pair内计算加权矩。以任意矩函数\(a_k\in\{r_k,d_k,r_kd_k,r_k^2,d_k^2\}\)为例：

\[
\overline a_{pnc}
=
\frac{\sum_k\omega_k^{\mathrm{stat}}a_k}
{\sum_k\omega_k^{\mathrm{stat}}+\varepsilon}.
\]

随后在同一target profile-cell内对不同neighbor profile等权平均：

\[
\overline a_{pc}
=
\frac{1}{N_{pc}}\sum_{n=1}^{N_{pc}}\overline a_{pnc}.
\]

最后以target profile为独立统计单位跨profile平均：

\[
\mu_{a,c}
=
\frac{1}{P_c}\sum_{p=1}^{P_c}\overline a_{pc}.
\]

单元协方差、方差和相关系数由这些最终profile-level矩构造：

\[
\operatorname{Cov}_c=\mu_{rd,c}-\mu_{r,c}\mu_{d,c},
\]

\[
V_{r,c}=\mu_{r^2,c}-\mu_{r,c}^2,
\qquad
V_{d,c}=\mu_{d^2,c}-\mu_{d,c}^2,
\]

\[
\rho_c=
\frac{\operatorname{Cov}_c}
{\sqrt{V_{r,c}V_{d,c}}+\varepsilon}.
\]

bootstrap继续以UTC日期分层、以target profile为重采样单位，并在每次重采样中重新计算中心矩。训练中的经验相关性loss必须复现“token→target-neighbor profile pair→target profile-cell→跨target profile”的层级，不能用最终profile-level均值和方差直接标准化token残差或token协方差。

### 6.3 模型相关性监督

模型端点交叉协方差定义为：

\[
K_\theta(q,x_j)=\frac{1}{7}y_q^Ty_j
=F(q)^TF(x_j).
\]

同时定义模型误差方向相关性：

\[
R_\theta(q,x_j)=e(q)^Te(x_j).
\]

\(K_\theta\)和\(R_\theta\)的训练估计均使用与离线统计相同的三层聚合顺序。以协方差为例：

\[
\widehat K_{pnc}
=
\frac{\sum_k\omega_k^{\mathrm{stat}}K_\theta(q_k,x_{j,k})}
{\sum_k\omega_k^{\mathrm{stat}}+\varepsilon},
\]

\[
\widehat K_{pc}
=
\frac{1}{N_{pc}}\sum_n\widehat K_{pnc}.
\]

\(R_\theta\)以完全相同的权重和层级得到\(\widehat R_{pc}\)。训练以target profile-cell为基本样本：第一项将聚合后的模型协方差与经验协方差比较，并用冻结的profile-level方差作量纲归一化；第二项直接将聚合后的模型误差方向与经验相关系数比较。两项均不得在token层使用profile统计量标准化：

\[
L_{\mathrm{error\text{-}structure}}
=
\frac{1}{|\mathcal P|}
\sum_{(p,c)\in\mathcal P}
\left[
\operatorname{Huber}\!\left(
\frac{\widehat K_{pc}-\operatorname{Cov}_c}
{\sqrt{V_{r,c}V_{d,c}}+\varepsilon}
\right)
+
\operatorname{Huber}\!\left(\widehat R_{pc}-\rho_c\right)
\right].
\]

只有stable单元进入该loss。每个target profile-cell等权贡献，防止拥有更多token或neighbor profile的样本支配训练。

### 6.4 状态长度尺度校准

状态长度尺度按profile pair而非token校准。对每个有效\((p,n,c)\)分别定义方向距离和对数幅度距离：

\[
D_{pnc}^{2}
=
\frac{
\sum_km_k\lambda_{\mathrm{phys},k}
\|e(q_k)-e(x_{j,k})\|_2^2
}{
\sum_km_k\lambda_{\mathrm{phys},k}+\varepsilon
}.
\]

\[
A_{pnc}^{2}
=
\frac{
\sum_km_k\lambda_{\mathrm{phys},k}
\left[a(q_k)-a(x_{j,k})\right]^2
}{
\sum_km_k\lambda_{\mathrm{phys},k}+\varepsilon
}.
\]

每个profile pair只贡献一个\(D_{pnc}\)和一个\(A_{pnc}\)，并使用train-only中经验相关系数为正且stable的单元确定：

\[
\ell_e
=
\operatorname{median}
\left\{
D_{pnc}:\rho_{\mathrm{train},c}>0,\ c\ \text{stable}
\right\}.
\]

\[
\ell_a
=
\operatorname{median}
\left\{
A_{pnc}:\rho_{\mathrm{train},c}>0,\ c\ \text{stable}
\right\}.
\]

校准至少需要1000个有效profile pair；\(\ell_e\)或\(\ell_a\)非有限、非正或样本不足时不得启用状态核。训练开始前固定校准profile pair及其条件单元，随后在每个Analysis epoch结束时用当前\(e\)和\(a\)重新计算两个尺度，并从下一epoch开始使用。这样长度尺度可跟随表示演化，但样本身份不变。\(\ell_e\)与\(\ell_a\)随每个checkpoint保存并写入run manifest，不在development或推理阶段重新估计。

## 7. 防止低维状态塌缩

### 7.1 Precision加权Gram whitening

将所有有效观测端点的7维因子坐标记为\(F\)，使用生产ETKF当前实际precision构造：

\[
G=F^TWF,
\qquad
W=\operatorname{diag}(\pi_j).
\]

归一化Gram loss为：

\[
L_{\mathrm{Gram}}
=
\left\|
\frac{G}{\operatorname{tr}(G)+\varepsilon}
-\frac{I_7}{7}
\right\|_F^2.
\]

trace归一化消除整体放大异常幅度带来的伪改善。该loss用于抑制观测子空间共线性、提高\(HX\)有效秩，但不能替代以Background residual为依据的相关性监督。

### 7.2 条件保持的profile配对打乱

为了检验误差状态是否超越粗分层变量，构造profile级对照配对。打乱只能在以下完整条件块内部进行：

\[
(s_t,s_o,a_t,a_o,c_{LT},c_{u,\mathrm{phys}},
c_{\rho,\mathrm{legacy}},d_{\mathrm{UTC}}),
\]

其中依次表示target来源、observation来源、target高度层、observation高度层、地方时类别、M2-U物理邻近度层、legacy \(\rho\)层和UTC日期。加入\(c_{u,\mathrm{phys}}\)可避免打乱对照因新物理核权重不同而产生伪差异；legacy距离则继续控制相对于M2-O几何邻域的可比性：

\[
\rho_{\mathrm{legacy}}
=
\max\!\left(
\frac{|\Delta\mathrm{lat}|}{5^\circ},
\frac{|\Delta\mathrm{lon}|}{15^\circ},
\frac{|\Delta t|}{1.5\ \mathrm{h}}
\right),
\]

并为\(\rho_{\mathrm{legacy}}\ge1\)设置明确的溢出层，避免新宽支持候选被错误截断到旧`0.75–1`单元。legacy距离不参与M2-U precision或新representativeness插值。整个neighbor profile及其高度token作为一个整体被置换，不能在token层独立打乱。若跨条件块打乱，模型可能仅凭来源、高度、昼夜或日期信息区分正负样本，不能证明其学习了更细的背景误差关系。

配对对照用于组合激活检查：匹配profile pair的误差状态相关结构应优于同条件块内的打乱配对。Kp/F10.7历史扰动作为独立敏感性诊断，不与profile配对打乱混为同一统计量。

### 7.3 增益响应一致性

增益响应约束使用两类query pair。第一类为局部连续性pair：对query \(q\)构造小幅四维扰动\(q'_{\mathrm{local}}\)。两者使用相同的train-only白名单和相同的目标profile排除ID；若正向扰动超出月份或纬度边界，则使用等幅反向扰动，不将坐标直接截断到边界点。

第二类为误差状态pair。定义尺度归一化联合距离：

\[
d_\eta^2(q,q')
=
\frac{\|e(q)-e(q')\|_2^2}{\ell_e^2}
+
\frac{[a(q)-a(q')]^2}{\ell_a^2}.
\]

在固定train-only query池中，依据当前epoch末stop-gradient的\(\eta\)选择\(d_\eta(q,q'_{\mathrm{state}})\le1\)的状态近邻，并要求两者具有非零共同物理观测支持。该pair不要求坐标相邻，用于直接检验低维误差状态相似性。若两query分别来自不同目标profile，则二者共同排除两个目标profile ID，保证两次求解使用完全相同且无自观测泄漏的排除集合。近邻检索仅决定训练配对，不形成跨query共享分析状态。

首个Analysis epoch尚无上一epoch的状态近邻表，因此只计算局部连续性pair；该epoch结束后在固定query池上生成状态近邻表，从下一epoch开始计算状态pair loss。状态pair loss可在\(\lambda_{\mathrm{state}}=1\)阶段提前训练误差表示，不等同于提前启用状态局地化核。

令\(\mathcal J=\mathcal J(q)\cup\mathcal J(q')\)为候选profile并集。对并集中的每个实际token，分别重新计算\(q\)和\(q'\)对应的physical、state、representativeness、\(C\)、\(A\)和\(G\)，不能复用另一query的权重或系统矩阵。增益响应loss为：

\[
L_{\mathrm{gain}}
=
\frac{
\sum_{j\in\mathcal J}
\overline\omega_j
\,\operatorname{Huber}\!\left(G(q,x_j)-G(q',x_j)\right)
}{
\sum_{j\in\mathcal J}\overline\omega_j+\varepsilon
},
\]

为避免状态核通过主动减小自身权重来掩盖响应差异，比较权重不含\(\lambda_{\mathrm{state}}\)。先定义不含状态核的基础precision：

\[
\pi^{\mathrm{base}}_{s,j}(q)
=
\frac{m_{s,j}\lambda_{\mathrm{phys},j}(q)
\lambda_{\mathrm{rep},s,j}(q)}{R_s},
\]

再令：

\[
\overline\omega_j
=
\sqrt{\pi^{\mathrm{base}}_j(q)
\pi^{\mathrm{base}}_j(q')}.
\]

因此，只在两query均具有可比较物理观测支持时约束增益，但状态核本身不能关闭该约束。分别对局部连续性pair和误差状态pair计算\(L_{\mathrm{gain}}^{\mathrm{local}}\)与\(L_{\mathrm{gain}}^{\mathrm{state}}\)，并采用等权组合：

\[
L_{\mathrm{gain}}
=
\tfrac12L_{\mathrm{gain}}^{\mathrm{local}}
+\tfrac12L_{\mathrm{gain}}^{\mathrm{state}}.
\]

该loss约束同一观测innovation的响应规律，不加入\(\delta(q)=\delta(q')\)或增量相等loss。

总的表示学习目标为：

\[
L
=L_{\mathrm{analysis}}
+\alpha_{\mathrm{error}}L_{\mathrm{error\text{-}structure}}
+\alpha_{\mathrm{Gram}}L_{\mathrm{Gram}}
+\alpha_{\mathrm{gain}}L_{\mathrm{gain}},
\]

其中各权重沿用M2-O现有梯度比例校准思想确定，不以development结果反向调参。

## 8. 状态核启用时序

训练开始时先启用连续物理核、M2-U新representativeness统计、误差结构loss和Gram whitening，状态核暂取\(\lambda_{\mathrm{state}}=1\)。每个epoch结束后只执行一个组合激活检查。令\(\gamma_i\)为precision加权Gram矩阵的非负特征值，\(p_i=\gamma_i/(\sum_r\gamma_r+\varepsilon)\)，有效秩定义为：

\[
r_{\mathrm{eff}}=\exp\!\left(-\sum_{i=1}^{7}p_i\log(p_i+\varepsilon)\right).
\]

组合检查内容为：

1. \(F\)、\(e\)、\(a\)、Gram特征值、\(\ell_e\)及\(\ell_a\)全部有限，两个尺度均由同一组不少于1000个有效profile pair校准；
2. 固定train-only校准样本上的联合\(r_{\mathrm{eff}}\)中位数不低于2.0，FY和COSMIC单源中位数均不低于1.5；
3. 在完整条件块内，匹配profile pair的\(R_\theta\)经验相关误差低于打乱配对：

\[
E_{\mathrm{matched}}
=
\operatorname{mean}_{(p,c)}
\operatorname{Huber}(\widehat R_{pc}-\rho_c)
<
E_{\mathrm{shuffled}},
\]

其中\(E_{\mathrm{shuffled}}\)使用同一target profile和条件单元、仅置换neighbor profile后按相同层级重新计算。

若完整编号epoch \(k\)末首次通过组合检查，则从epoch \(k+1\)开始启用状态核，并将首次实际使用状态核的epoch \(k+1\)记录为`state_localization_activated_epoch`。状态核一旦启用，在同一训练中不再关闭；后续epoch仍更新\(\ell_e\)与\(\ell_a\)，但不再重复决定是否关闭状态核。若截至完整编号epoch 15仍没有任何epoch实际使用状态核，则该run只能标记为“连续物理局地化消融”，不能作为M2-U状态依赖局地化结果解释。

为避免增加多重阶段门禁，M2-U仅保留上述一个训练内组合激活检查，以及训练完成后的development综合评价。其他有效秩、连续性、方向率和RMSE均作为同一评价报告中的诊断量，而不形成逐级中止链。

## 9. 当前代码差异与最小改造范围

M2-U不重写FSIA-INR主干，只修改观测检索、局地权重、源项累积及相应训练监督。当前代码与目标架构的对应关系如下。

| 现有模块 | 当前M2-O事实 | M2-U最小改造 | 保持不变的语义 |
|---|---|---|---|
| `inr_modules/config_mdia.py` | 使用\(\pm1.5\) h、\(\pm5^\circ\)纬度、\(\pm15^\circ\)经度、top-8 profile和每profile 8个高度token | 在独立M2-U配置中增加`localization_semantics=continuous_error_state`、\(L_s\)、\(L_t\)、状态核、统计文件和激活记录；不改M2-O默认配置 | D64/N8、Global R、Background seed、日期阻断 |
| `inr_modules/data_managers/FY_dataloader.py` | FY/COSMIC分别返回固定\([B,8,8]\)候选；先矩形筛选，再按归一化L1距离取top-8；payload使用max-norm `rho_squared` | 复用时间分箱、profile ID、预采样高度token和白名单逻辑；新增无hard top-k的安全粗筛与确定性分块迭代，分别保留新\(\lambda_{\mathrm{phys}}\)和legacy \(\rho\) | 每profile最多8个有效token、同来源目标profile排除、FY/COSMIC统一payload语义 |
| `inr_modules/mdia/sliding_dataset.py` | representativeness从旧top-8统计文件加载，并按旧max-norm \(\rho\)单元连续插值 | 加载M2-U新schema和输入SHA256；改按\(u_{\mathrm{phys}}\)插值并继续使用floor 0.25，拒绝旧统计身份 | Background端点计算、train-only白名单 |
| `inr_modules/mdia/fsia_model.py` | `_localization_precision`由max-norm `rho_squared`生成；`_source_terms`一次物化\(Y\)、\(C\)和\(b\) | precision直接接收物理核×representativeness×状态核；候选按确定性profile chunk累积\(C_s\)、\(b_s\)及必要诊断，chunk仅是计算分区，不是观测选择 | endpoint-context basis、`observation_factor_coordinates`、Cholesky求解、M00/M10/M01/M11 |
| `inr_modules/mdia/train_fsia.py` | Gram loss已有可选实现，但输入为完整token因子张量；经验协方差loss在token层匹配冻结单元目标；架构签名尚不记录局地化语义 | 复用Gram公式、exact-mode权重和profile平衡语义，将Gram改为chunk级矩阵累积；替换为profile-blocked误差结构loss，增加条件保持打乱、\(\ell_e/\ell_a\)校准和增益响应loss；升级checkpoint格式并严格保存/校验局地化语义 | exact mode loss、Global R、梯度比例校准思想 |
| `main_fsia.py`与run manifest | 当前路径和恢复检查面向M2-O representativeness及架构签名 | 将M2-U统计NPZ/JSON、物理核参数、固定校准query/profile pair身份和代码起点写入输入身份；拒绝缺失或不一致的M2-U文件 | QC report与NPY/NPZ SHA256审计、日期隔离 |
| `evaluate_satellite_development.py` | 已计算M00/M10/M01/M11、方向、负响应、RMSE及hard invariants，但归因诊断依赖完整token数组 | 将innovation、协方差、gain符号和谱指标改为chunk级计数/矩累积；增加候选切换、增量连续性、状态匹配—打乱诊断 | 现有development门禁阈值与locked-test隔离 |

变量候选不能通过简单增大`*_nb_k_prof`实现，因为任何有限top-k仍会产生成员替换接口。候选必须遍历物理正支持内的全部profile。为控制内存，按固定profile ID顺序分块计算端点basis并累积充分统计量；不同chunk的求和应与一次性物化全部观测在数值容差内一致。现有Gram函数和development归因函数不能原样接收该数据流，只复用其数学定义、profile平衡和门禁语义。训练与评价所需逐token指标应在chunk内转换为计数、加权和、Gram矩或profile-level矩后再归并，不建立新的全域观测张量。

分块减少观测张量物化，但普通autograd仍可能保留各chunk的density-basis中间激活。首版先缩小Analysis query batch并记录峰值内存；若仍超出资源，再对共享density-basis调用PyTorch原生gradient checkpointing。该措施只改变反向重计算，不改变候选集合、充分统计量或ETKF数学结果。

M2-U必须升级checkpoint格式版本。架构签名至少新增`localization_semantics`、物理核类型、\(L_s\)、\(L_t\)、状态核floor、方向—幅度误差状态定义和M2-U统计schema版本；运行状态另存当前\(\ell_e\)、\(\ell_a\)、状态核是否已启用及`state_localization_activated_epoch`。恢复训练或development加载时，这些字段、统计文件SHA256和固定校准样本身份必须完全一致。旧M2-O/Gram checkpoint缺少上述字段时只能按M2-O语义加载，不能静默补默认值后作为M2-U运行。

## 10. 实施顺序

### 10.1 版本隔离

在修改模型前，先将当前ISR未提交改动作为独立WIP提交并推送当前分支；报告目录继续排除在Git之外。为生产M2-O恢复点`c31cb36`建立只读标签。M2-U不能直接从该标签创建分支，因为`c31cb36`不包含后续的Gram whitening与paired-loss兼容提交。独立worktree应明确从`cec9de8`创建；该提交包含`16cf22a`与`cec9de8`两项M2-T基础修改，但不包含随后单独提交的ISR WIP：

`D:\code11\IRI01\IRI03\INR1-1-run65-m2u`

并创建分支：

`codex/run66-m2u-continuous-response-etkf`

所有M2-U修改、辅助统计和轻量测试均在独立worktree完成，不读取或干扰正在运行的M2-O训练目录。`c31cb36`只用于严格恢复原生产M2-O，`cec9de8`才是M2-U的代码起点；两者不得混用。

### 10.2 第一阶段：连续物理局地化骨架

首先复用现有profile index和每profile八点垂直采样，增加宽支持候选检索、profile包络粗筛和实际token精确权重。ETKF不再接收hard top-8后的观测矩阵，而对全部正权重token累积\(C_s\)和\(b_s\)。该阶段先确保M00 fallback、M10/M01独立系统和M11联合系统保持不变。

### 10.3 第二阶段：M2-U辅助统计

在train-only日期上以冻结Background重新生成M2-U独立统计。新统计严格使用\(m\lambda_{\mathrm{phys}}\)，保持profile-blocked三层聚合和日期分层bootstrap，并保存完整输入身份。旧M2-O统计保留用于历史复现，但M2-U配置必须拒绝加载旧schema或旧SHA256。

### 10.4 第三阶段：误差状态核与表示学习

复用`observation_factor_coordinates`得到\(F\)和\(e\)，加入状态长度尺度校准、带floor的状态核、分层经验相关性loss、条件保持的profile打乱诊断，以及query pair增益响应loss。状态核进入precision时停止梯度，相关性与Gram loss负责改善表示，避免局地化核与表示之间形成自我强化的捷径。

### 10.5 第四阶段：训练与development评价

M2-U训练继续复用冻结Background并执行10个Analysis epoch。训练内只使用一个组合激活检查。训练完成后统一评价：FY/COSMIC profile RMSE、M00/M10/M01/M11精确语义、方向与负响应、\(HX\)有效秩、连续性切片、相邻query增益响应差异，以及状态相似query的响应一致性。只有精度、方向和连续性同时改善时，M2-U才可替代M2-O；图像更平滑本身不构成接受依据。

### 10.6 后续升级条件

若连续候选、状态核和增益响应loss仍不能形成足够平滑且可泛化的响应，再考虑重叠低维基函数系数ETKF：

\[
\delta(q)=\sum_ra_r\psi_r(e(q)).
\]

其中局地重叠基函数\(\psi_r\)使相似状态共享部分分析系数。该方案只在M2-U首版证据不足时启动，不预先实现，也不恢复M2-P的单一统一latent。

## 11. 最小验证与验收

实现阶段只保留能够捕获语义错误的最小测试：

1. 候选profile在零权重边界进入或离开时，\(C\)、\(b\)和输出不发生跳变；
2. 无正权重观测时严格退回M00；M10/M01独立求解，M11联合求解；
3. 合成数据验证统计层级为token→profile pair→target profile-cell→跨target profile，重复token或增加同一profile内点数不能改变profile权重；
4. \(q'\)边界反向扰动、候选并集及白名单/排除ID具有确定性；
5. 新统计schema、SHA256、状态核激活epoch和checkpoint恢复信息完整一致。

最终development报告至少同时给出分析精度、响应方向、观测子空间有效秩与条件数、空间/时间连续性，以及匹配—打乱profile pair差异。locked-test与ISR只在M2-U模型冻结后按既定顺序使用。

## 12. 期望结果与判定方式

### 12.1 结构保持结果

M2-U首先应保持M2-O已经确认的正确语义。在相同Background checkpoint和输入坐标下，M00应与M2-O逐元素一致；无正precision观测时M10、M01和M11均严格退回M00；M10和M01仍为独立单源系统，M11仍为一次联合系统。连续局地化只改变哪些观测以何种precision进入\(C_s\)和\(b_s\)，不能改变raw innovation定义、Global R或Cholesky ETKF求解。

### 12.2 连续性结果

对固定development坐标构造小幅空间与时间扰动，并特别抽取M2-O top-8成员发生切换的位置。定义分析增量差：

\[
D_\delta(q,q')=|\delta(q')-\delta(q)|,
\]

以及共同候选上的加权增益响应差：

\[
D_G(q,q')=
\left[
\frac{\sum_{j\in\mathcal J}\overline\omega_j
\left(G(q,x_j)-G(q',x_j)\right)^2}
{\sum_{j\in\mathcal J}\overline\omega_j+\varepsilon}
\right]^{1/2}.
\]

期望M2-U在相同扰动尺度下显著降低\(D_\delta\)和\(D_G\)的中位数、95%分位数与99%分位数，且改善集中出现在原top-8切换边界，而不是依靠整体压小分析增量获得。为排除这种伪改善，报告还须同时给出增量RMS、precision总质量和有效观测profile数量。

### 12.3 误差状态与观测子空间结果

冻结M2-O历史诊断中，\(\operatorname{rank}(X)\approx6.97\)，但\(HX\)有效秩约为1.58，条件数约为\(1.7\times10^4\)。M2-U期望提高precision加权\(HX\)的有效秩并降低条件数，同时使匹配profile pair的误差结构预测优于完整条件块内的打乱配对。

增益响应结果分别报告局部连续性pair和非局部状态近邻pair。对状态近邻pair，还需构造来源、高度、地方时、\(u_{\mathrm{phys}}\)、legacy \(\rho\)、UTC日期和共同观测质量相匹配、但\(d_\eta>1\)的对照pair。期望满足：

\[
\operatorname{median}D_G(\mathcal P_{\mathrm{state}})
<
\operatorname{median}D_G(\mathcal P_{\mathrm{control}}).
\]

该比较才直接检验“低维误差状态相似的query具有相似观测响应”。状态相似query不要求在不同innovation下获得相同\(\delta\)。

状态核首次启用的最低结构条件已在第8节预注册；最终报告不以“通过激活”替代结果分析，而应将M2-U与M2-O在相同development日期、query和观测白名单上成对比较。若有效秩上升但协方差符号或RMSE恶化，则不能解释为成功。

### 12.4 精度与方向结果

M2-U沿用现有development最终门禁，不额外建立逐阶段中止链：FY与COSMIC自源负响应均不低于70%，两个跨源总体方向率均不低于60%，所有可估计的M11高度×昼夜×纬度单元方向率均不低于55%，各来源自源profile RMSE相对M00恶化不超过1%，M11相对最佳单源恶化不超过1%，且`passed_hard_invariants=true`。

预期M2-U在不破坏上述精度与方向条件的前提下减少局地接口，并改善\(HX\)可辨识性。状态依赖局地化不能保证每个query或每条profile都优于M00，也不能把错误innovation变为正确信息；若连续性改善仅来自增量趋近零，或innovation/协方差符号仍系统错误，则应保留M2-O并将M2-U判定为未达到研究目标。

## 13. 最终核查结论

本计划已完成最终一致性修正：明确M2-U必须重新生成统一的representativeness与经验协方差辅助统计，并消除统计权重循环依赖；离线统计和训练loss严格保持profile-blocked层级；profile配对打乱保持完整条件结构；误差状态同时包含归一化方向与对数幅度，两个长度尺度按固定train-only profile pair校准并按epoch更新；增益loss同时覆盖局部扰动pair和非局部状态近邻pair，且比较权重排除状态核以防自我屏蔽；状态核按“epoch末检查、下一epoch实际启用、启用后不关闭”运行；物理先验固定为1800 km球面水平核×1.5 h时间核，不再用人工高度紧支撑重复规定垂直相关。

结合当前代码核对后，必须修改的位置限定为候选检索、payload局地权重、ETKF源项累积、辅助统计、训练监督、checkpoint语义和development分块归因；M00、D64/N8正交因子、endpoint-context basis、Global R及精确M10/M01/M11可直接复用。现有Gram与归因函数只能复用数学语义，不能原样用于变量候选。M2-U代码起点修正为包含Gram提交的`cec9de8`，`c31cb36`仅保留为生产M2-O恢复点。公式上，物理核、状态核和representativeness均以非负precision乘子进入ETKF，\(A(q)\succeq7I\)保证求解正定，增益响应公式对应分析增量对raw innovation的偏导。

计划未引入oracle innovation裁剪、推理期监督、单一全域latent或跨query分析状态融合。最终核查未发现尚未处理的ETKF维度或统计层级矛盾；主要剩余风险是无hard top-k后的候选计算成本、误差状态仍可能塌缩，以及背景残差相关不等同于纯背景误差相关。三者已分别通过确定性分块充分统计量、组合激活检查和结果解释边界予以约束，但只有实际train-only统计与development结果能够判定M2-U是否成立。
