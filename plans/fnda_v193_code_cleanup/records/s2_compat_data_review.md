# S2 数据与检查点兼容性审查

身份：S2 的只读专项证据，供总体依赖图及保留/简化/归档清单合并使用。2026-10-09。

按用户最新确认，当前工作区是 N8D64 参数敏感性重跑使用且参数与结果一致的训练源码。本审查不重跑该实验，也不加载权重或未知 pickle。用户确认与本轮静态代码核查分别记录。

## 必须保留的执行关系

| 对象 | 证据与用途 | S2 分类 |
|---|---|---|
| `inr_modules/mdia/checkpoint_io.py` | `load_fsia_analysis_checkpoint` 第95行起读取权重旁的 `run_manifest.json` 和 `training_summary.json`，依域版本核实配置；第279、283行以 `weights_only=True` 读取 IRI/主模型，288行 `strict=True` 加载；拒绝非有限权重；由绘图、GIRO、ISR等使用 | 保留加载及身份检查，不以“旧版本”删除 v14 |
| `inr_modules/mdia/v14_contract.py` | 提供 v14 域/低高度 IRI 约束/训练协议/Gram 校准，入口和训练、加载器共同导入；v14 是主权重实际格式 | 保留；v15 A2S1 分支列为依赖范围待审，不能整文件移除 |
| `inr_modules/data_managers/irinc_neural_proxy.py` | IRINeuralProxy 经 `data_managers/__init__.py` 导出；加载器固定构造 `[4,128,128,128,128,1]`；`fsia_model.py:818-819` 注册代理子模块并冻结 | 保留类名、层结构、forward 特征接口及冻结逻辑；其独立训练入口是 IRI 复现工具，不能当成未用主训练分支删除 |
| `inr_modules/data_managers/FY_dataloader.py` | `train_fsia.py:119-124` 导入两类数据集及 FY/COSMIC 邻域索引；567行起 NPY mmap/回退读取，584行起 profile index/元数据对齐校验；COSMICDataset 在1595行继承 FY3D_Dataset | 保留数据列顺序、剖面ID、索引边界、观测筛选及邻域行为；不能以继承复用认定 COSMIC 类多余 |
| `inr_modules/data_managers/profile_split.py` | 训练可选 profile split；`checkpoint_io.py:332` full30 allowlist；`evaluate_giro_peak.py`、`isr_evaluation/main_isr_eval.py:1618` 与 `traditional_static_etkf/runner.py:81` 共用 | 保留论文共享样本评估链路；原主训练采用日期划分不构成删除理由 |
| `inr_modules/data_managers/date_split.py` | 严格日期清单读取、转换、导出及兼容别名；主训练记录为20 train/5 dev/5 locked-test | 保留主日期清单解析；A1S6转换与旧别名仅在全调用清单和保留评估范围确定后考虑归档/简化 |
| `inr_modules/data_managers/space_weather_manager.py` | `train_fsia.py:7808` 构建；读取 OMNI 驱动 | 保留 |
| `inr_modules/data_managers/iri_peak_manager.py` | `train_fsia.py:7073-7074` 局部导入并构建；读取 hmF2/NmF2 NPY，插值支持峰值约束 | 保留；函数内导入不属于无依赖模块 |
| `inr_modules/density_units.py` | 统一 m^-3 与 10^11 m^-3 显示；被 IRI、模型评估、GIRO、ISR/可视化引用 | 保留单位转换与 NaN 语义 |
| `qc_profile_data.py` | `test_qc_profiles.py` 验证 QC 输出与 FY profile index边界 | 保留为预处理复现候选，需与父清单的具体数据来源关联 |

## 序列化与配置覆盖风险

1. 主分析权重是模型 state_dict。`fsia_model.py` 将 IRI 注册为子模块，且包括 `r_fy/r_cosmic` 等 buffers 与配置条件下创建的网络；重命名属性或将子模块挪出模型可能改变键名，严格加载兼容性必须在 S3 固定。
2. 训练续训文件是另一种格式：`train_fsia.py:8099` 使用 `weights_only=False`，8233行读取 `model_state_dict`；8970行附近保存模型、optimizer、scheduler、scaler 与随机数状态。不能把所有 `.pth` 都按分析权重处理，也不能为代码整洁统一删除续训状态/校验。本轮未执行该加载。
3. `checkpoint_io.py` 同时支持格式12/13/14/15，v14主模型不能随旧分支归档。格式12/13和A2S1分支可列为简化候选，但须由全局调用/保留比较实验决定；尚无删除结论。
4. 主训练 manifest 的 `config.save_dir` 保留背景阶段路径，实际 Analysis配置/续训覆盖另有记录；不能用当前默认配置或 manifest单个字段覆盖最终运行身份。严格加载器目前要求 basis_dim=64、enkf_n_members=8，故敏感性非N8D64加载路径也要由总体调用图分别记录，不能误认为全项目统一入口。
5. `verify_hashes`、`allow_historical_epoch`、`expected_sha256` 属于明示的验证/历史epoch控制，不应直接删除；必须逐调用检查是否服务论文缓存和历史评估。
6. `iri_peak_manager.py` 的 `__main__` 自测含 `sys.path` 与 `__import__('torch')`。可作为低风险自测代码简化候选，但先确认命令使用方式；没有证据支持删除运行时 IRIPeakManager。

## 数据身份与不可混淆项

主训练实际输入为 FY QC `.npy`+剖面索引`.npz`+QC报告、COSMIC QC `.npy`+索引`.npz`+报告、IRI代理权重、IRI hmF2/NmF2格点、OMNI文件、日期划分清单。S1已记录11项输入身份；本轮只核对路径及消费代码。

- FY：`D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy`，配套 index/report。
- COSMIC：`D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc.npy`，配套 index/report。
- IRI：`D:\code11\IRI01\output_results\iri_september_full_proxy.pth`。
- 公开发布的 `cosmic_qc.npz` 是归档产品，不等于上述训练 `.npy` 的字节身份；不要替换训练路径来“统一”文件组织。
- FY/COSMIC 最前五列用于坐标/目标，COSMIC额外元数据与FY索引/磁坐标sidecar分别有读取契约；移动/裁剪列或改变dtype可能改变筛选和邻域结果。

## S3建议固定的最小基线

1. 记录原分析权重的完整state_dict键名、shape、dtype及哈希；IRI、配置、manifest/summary和输入路径一起固定。仅在明确可信本地权重上执行加载。
2. 固定小批次坐标、空间天气、邻域输入，记录背景/无观测回退/有观测分析输出，以及 FY/COSMIC profile IDs 与索引选择；比较清理前后完全相同输入。
3. 保留并按改动选择现有测试：`test_resume_schedule.py`、`test_m2w_domain.py`、`test_v14_gram_calibration.py`、`test_v14_fixed_query_regression.py`、`test_qc_profiles.py`、`test_qcv2_background_contract.py`、`test_profile_split_a1s2.py`、`test_density_plot_units.py`、`test_giro_shared_full30.py`、`traditional_static_etkf/test_runner.py`。A2S1/A1S6及磁坐标sidecar测试随对应保留范围决定去向，不先删测试来让收敛后的实现通过。
4. 源码备份须包括函数内导入目标和预处理工具；数据/检查点先记录路径/哈希，不在 S2 复制大文件或写正式输出目录。

## 本轮限制

仅静态读取，没有运行权重加载、数据处理、测试或训练。全仓文件枚举遇到若干历史pytest临时目录访问拒绝，不影响上述明确路径读取；它们的内容与可删除性尚未检查。本专项无可以立即删除的文件结论，简化候选需合并全局调用与论文保留范围后进入S3/S4。
