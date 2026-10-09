# S1 模型身份与结果溯源报告

日期：2026-10-09。身份状态：**TRAINING_SOURCE_CONFIRMED_BY_AUTHOR**；阶段进度见 [总体计划](../MASTER_PLAN.md)。

补充确认：用户在 S2 启动前说明，当前工作区已在参数敏感性实验中重新运行 N8D64，所有参数与结果均一致，已证实为训练源码。本项目据此将当前工作区认定为已确认的 N8D64 训练源码。以下历史冻结快照及缓存生成时哈希缺口仅作来源记录，不再作为否定训练源码身份的理由；本轮没有重新执行该训练实验。

[总体计划](../MASTER_PLAN.md#阶段与关系) · [S1 路线](../01_MODEL_IDENTITY.md) · [本轮决策](decisions.md) · [机器可读证据](model_identity.json) · [S2 路线](../02_DEPENDENCIES_SCOPE.md)

## 结论与采用的身份

S1 身份核查完成。用户进一步确认当前工作区通过 N8D64 敏感性重跑验证，作为训练源码和后续清理基线。论文主模型沿用此前作者指定的 A1S/N8D64 权重；历史论文预测缓存单独保留，不倒填生成时未记录的哈希。

以下路径除特别说明外，均相对 `D:\code11\IRI01\IRI03\INR1-1-run65`。绝对路径、现时 SHA-256、预期哈希和匹配结果均见 JSON。

| 身份 | 对象 | 判定 |
| --- | --- | --- |
| 历史论文主 run | `checkpoints_fsia/run67-m2w-120-500-analysis-a1s-10epoch` | ISR candidate 与 GIRO baseline 的历史路径共同指向此 run |
| 当前指定复现权重 | 上述目录 `best_fsia_model.pth` | 与作者指定 N8D64 归档权重字节一致 |
| 主背景 | `checkpoints_fsia/run67-m2w-120-500-background-stratum-bias-a1s/best_background_model.pth` | 与主 Analysis run 内 background 文件一致 |
| 已确认训练源码／维护代码 | 当前工作区，38 个已核对相关文件 | 用户确认 N8D64 重跑参数与结果一致；38 文件哈希与 9 月账本一致 |
| 论文数值依据 | 正式 ISR/GIRO、独立 toggle、支持度和敏感性等保存结果 | 通过来源映射保留各自身份，不能混用 |

## 权重、配置与训练记录

主权重 SHA-256：`94f4da0b265c313a812159560db25c5ae5394b29603d75f47b18d5f6d9bb6dc1`，579315 字节。其归档副本为 `FNDA_V19_slim_package/B_model_code/weights/sensitivity/N8D64_reproduction_best_fsia_model.pth`，本次重新计算哈希一致。

主背景 SHA-256：`2ff04fa804fdc703ee741378dcfc0caa06147b616a812d64e848e58245c0cef9`。IRI 代理权重为 `D:\code11\IRI01\output_results\iri_september_full_proxy.pth`，SHA-256：`e567c2a922ace0fb0ada59e12946bc3c2e799c336005fc5f90244c71c7bdcd31`。

| 项目 | 记录值 |
| --- | --- |
| 训练阶段 | Background 5 epochs + Analysis 10 epochs |
| 选中轮次 | Background epoch 3；Analysis global epoch 15 |
| 种子 | seed=42、analysis_seed=42 |
| 模型规格 | D=64、N=8；checkpoint format 14 |
| 密度基函数语义 | `endpoint_context_symmetric` |
| 分析状态语义 | `legacy_feature_increment` |
| 上下文／模态基 | `query_conditioning` / `learned_density_basis` |
| 模型／观测／峰值搜索高度 | 120–500 / 200–500 / 200–500 km |
| 低高度协议 | `soft_iri_background_zero_analysis_increment_v1` |
| 同化语义 | `continuous_physical_local_letkf` |
| 空间／时间局地化 | 1800 km / 1.5 hours |
| 背景 trust gate | disabled |
| Analysis 训练 | exact mode loss；observation Gram loss 开启；经验协方差及方向损失关闭 |

依据为主 run 的 `run_manifest.json`、`training_summary.json`、`fsia_training_history.json`、`training.log`；本轮记录文件身份，未执行模型。完整配置和 resolved_training 已写入 JSON。

**初始配置与最终运行必须分开。**主 manifest 的初始 `config.save_dir` 仍指向 Background run，`background_only=true`；但 resume event 的目标为 Analysis，训练历史共 15 轮，summary 为完成 Analysis。当前默认配置的 save_dir 又是 `run66-m2v-qcv2-dateblocked-background`。这些单项都不能作为完整的 V19.3 Analysis 运行配方。

当前加载入口 `inr_modules/mdia/checkpoint_io.py` 的 `load_fsia_analysis_checkpoint` 读取权重同目录 manifest/summary；S2 应追踪这套解析关系，保留必要配置覆盖和检查。S1 不修改历史 manifest，也不生成未经验证的“精简训练命令”。

## 训练输入与预处理身份

训练 manifest 的 11 项 data_identity 均已重新计算 SHA-256并匹配：FY QC 数据、索引、QC 报告；COSMIC QC 数据、索引、QC 报告；IRI 代理权重；IRI hmF2/NmF2 格点；OMNI 驱动；日期划分清单。

FY 主输入为 `D:\FYsatellite\EDP_data\fy_202409_qc_v2.npy`；COSMIC 主输入为 `D:\cosmic2\cosmic245-274-September\cosmic_september_2024_qc.npy`。这些是预处理输入，不能与官方原始观测或压缩归档文件混为同一字节对象。QC 版本和结果由输入哈希及报告固定；完整原始数据到 QC 的代码调用关系属于 S2 范围。

日期划分来自 `isr_validation_outputs/run66-modelonly-m0/date_split_manifest.json`：20 个训练日、5 个 development 日、5 个 locked-test 日，完整的零基日期索引见 JSON。训练记录含 FY 44625、COSMIC 56099 个训练剖面。训练环境记录为 Python 3.12.12、PyTorch 2.12.1+cpu；这是历史运行环境，不是当前环境重新验证结论。

## 当前代码身份及历史差异

HEAD：`b8caf0df7d860b512f7a32b0b93b19552e02674d`，提交说明 `Freeze A1-S background bias baseline`。当前有 14 个 tracked 文件修改，工作区相对 HEAD 的 diff 为 9241 行插入、610 行删除；HEAD 不能代表当前代码全部内容。

逐项核对 `FNDA_V19_slim_work/code_build_ledger.json` 中的 38 项原始源码哈希，38/38 与当前一致。核心文件现时身份包括：

- `inr_modules/mdia/fsia_model.py`：`f38a738fac1c924e46442aa743ad6cb5359acb6a60672858aed94230540bd085`。
- `inr_modules/mdia/train_fsia.py`：`ff35d806c8d05d5a550a7d7f4aca01a39d508bf51bc18fa86cb5c09955c20b60`。
- `main_fsia.py`：`bb83ab00dff9b85a5deb64f1a4767dc6f9a0001687f2c55b6caf2fcecd3e72b3`。
- `inr_modules/config_mdia.py`：`6dc74aef86bf878abd731b73381fa96c8a30ed9c73033835a12fd8dc367e0ff5`。

主 run 初始 code_identity 记录 110 个 Python 文件，聚合哈希 `7ec4eb49a46e9b3c53850b2bf1be029f950bb3f9e1353bdf2da0d29062af240c`；2026-08-21 的续训记录为 309 个文件，聚合哈希 `5f600d01e06307b8ac4cd5c510224a38bcbad980a5eca71e824ffe418a217395`。记录证明代码集合身份不同，但不能据数量变化推断模型行为变化。

存在 S6 实验源码快照：`FNDA_A1S_V19_archive_review/02_code/_external/D/code11/IRI01/IRI03/INR1-1-run65-A1S-restored/build/a1s_submission_v8/run_20260914_021617/experiments/sensitivity/source_snapshot`。与当前维护树相比，模型和训练器中已有 background_adapter_gate、A1S7/8/9、natural background、night 等后续实现。S6 快照不能直接替代主 A1S 源码；这些分支也尚未被判定为可删除。

移植包的 `main_fsia.py`、配置与 checkpoint loader 包含路径及发布策略调整；移植包并不是比当前工作树更权威的训练源码。S1 未枚举全项目未追踪文件，38 项哈希不代表完整项目备份，S2/S3 仍须补全范围。

## 论文结果到来源的连接

`model_identity.json.result_lineage` 覆盖 23 项（Figure 1–7、S1–S6、Table 1–4、S1–S6），共 75 条来源／生成器记录，均现时存在且与历史账本哈希匹配；计数包含不同图表共享的重复来源引用。

| 结果族 | 主要来源与身份 |
| --- | --- |
| ISR 主结果 | `isr_validation_outputs/m2w-a1s-full30-analysis-isr/isr_evaluation_cache.npz` 的正式 A1S；static ETKF 来源为 `traditional-static-etkf-raw-iri-full30-isr/isr_etkf_cache.npz` |
| GIRO 主结果 | `m2w-a1s2-vs-a1s-giro-full30` 混合缓存中的 `a1s_hm_M11_`、`a1s_nm_M11_`；不得选 A1S2 candidate |
| GIRO 三方法统计 | V15 data 的 `giro_{nm,hm}_three_method_a1s_v13.npz` 与 `peak_statistics/` |
| RO toggle | 独立 paired_endpoints 实验及 V9 review_package 的 ro_toggle 统计；不能用主缓存 M11 替代 |
| 支持度与机制 | restored 工作树的 E1/E2 及 W3_mechanism，V15 support_strata_v14 |
| 剖面与空间切片 | V15 frozen_v12 的 figure7/figure8 数据及 V9 slice_arrays |
| 敏感性 Table S6 | N4D64、N8D32、N8D64_reproduction 各自的 run、预测及共同评估总体 |

上述 V9/V15 完整路径及每项生成器见 JSON，restored 根目录为 `D:\code11\IRI01\IRI03\INR1-1-run65-A1S-restored`，属于外部依赖，不在本轮修改范围。

N4D64 权重哈希 `ebf8bfef5fd920c1912ea72585298954316efed6c24bfbe91a24827ed68db0b8`，N8D32 为 `40d96fe4076f51bc53efd1512bc6fe2366b06f99e5536a776a823e584f7019c6`。原 S6 run 的 manifest、summary、权重已读取／计算身份，详见 `sensitivity_runs`，不能因为主配置为 N8D64 就将论文敏感性链路判成无用。

13 张 V19.3 图中 12 张与 V19 图身份相同。Figure 2 与 V19.2 修复版字节一致，V19.2 的 layout_changes/final_integrity 记录为仅移除外置标题。本轮核对了哈希和该历史记录，没有重新进行视觉验收。

## 保留的证据缺口与冲突

1. ISR contract 的 candidate SHA 为 null；GIRO contract 的 baseline SHA 为 null 且 hash_verification=false。历史记录指向主 run 路径，但不能证明生成历史预测时的权重就是当前字节。此前作者指定仍有效；缓存独立保留。
2. 未找到可完整绑定早期训练过程的单一冻结快照。用户已通过当前工作区 N8D64 重跑确认训练源码身份；快照缺口仅限制早期逐字节时间溯源。
3. `a1s_submission_v17/run_20260920_012527/sources/s1_trace.py` 第 26 行指向 A1S5-1 epoch15，同时自称 authoritative/formal A1S，并要求 endpoint_context_symmetric。当前 A1S5-1 manifest 与此不相容。该 trace 保留为冲突证据，不覆盖此前作者指定，也不在清理时当作主入口。
4. 配置包含初始背景状态、续训解析状态和当前默认值三个层次。S2 需定位有效路径，S3/S5 再验证可执行行为。

这些限制已按本轮用户决定纳入身份定义；不是补造历史信息，也不要求现在重训。

## S1 验收与 S2 建议

S1 已完成路径／哈希核查、模型配置和阶段身份梳理、结果来源映射、当前代码差异审查及决策记录。证据采集脚本 `collect_s1_identity.py` 仅读取模型／项目工件，写入本目录 JSON，内置主权重身份、38 文件匹配、文件读取稳定性及 Figure2 修复版一致性断言；已实际运行通过。

S1交付时建议进入S2，仅做调用关系与处置候选清单。用户随后已授权并完成该阶段，见 [S2清单](scope_inventory.md)；目前等待是否进入S3的决定。

未执行模型加载、前向、训练、重算论文指标、代码清理或备份；哈希匹配证明文件身份，不等于推理或训练复现通过。
