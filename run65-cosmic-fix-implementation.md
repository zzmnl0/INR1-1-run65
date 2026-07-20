# run65 COSMIC 同化分支修复记录

## 状态

2026-07-20 完成最小根因修复、真实单批反向、小样本过拟合和四模式功能验收。未修改原 run65 代码目录或 checkpoint；当前产物是功能验证用 smoke checkpoint，不是完整重训教师。

## 根因与数据流

正式 FY 主训练路径已通过 `SlidingWindowBatchProcessor` 查询并传入 FY/COSMIC 邻域。COSMIC two-pass 和独立 COSMIC 验证路径只传坐标、空间天气与 IRI 峰值，未查询或传入 `neighbors_feats_cosmic/has_obs_cosmic`，因此 COSMIC loss 无法监督观测分支。

`NeuralETKFLayer` 又将 `w_COSMIC` 先乘 FY 的 `has_obs`，再乘 `has_obs_cosmic`，错误屏蔽 COSMIC-only 查询。另有启动死锁：COSMIC `input_proj`、`H_COSMIC_w` 和 `proj_pre` COSMIC 列同时为零时，三者无法建立首批有效梯度。

## 代码修改

- `inr_modules/mdia/train_fsia.py`：新增一个复用正式 `COSMICNeighborhoodIndex.query_batch_np` 的局部查询函数；COSMIC two-pass 和 COSMIC 验证均传入正式邻域及 mask。
- `inr_modules/mdia/fsia_model.py`：`w_FY` 仅乘 `has_obs`，`w_COSMIC` 仅乘 `has_obs_cosmic`。
- 为 COSMIC `input_proj` 和 `proj_pre` COSMIC 列提供小幅非零启动；无局部观测时编码输出仍为零，不改变 IRI 退化语义。严格续训旧全零 checkpoint 时只重启该入口。
- 未修改对称 `±1.5 h` 查询、逐点局部分析、无邻域退化规则，也未实现全球共享分析状态。

## 单批与小样本验证

环境为 `pytorch_cpu`。严格加载旧 checkpoint 后，仅重启 COSMIC 入口，使用 32 个真实 COSMIC 点和正式局部邻域。Adam、学习率 `1e-2`，仅优化 COSMIC 编码器、`H_COSMIC_w`、`R_COSMIC_net` 和 `log_r_ref_COSMIC`，共 40 步。

| 项目 | 结果 |
|---|---:|
| 初始/最终 MSE | 0.03921994 / 0.01113983 |
| `cosmic_obs_encoder.input_proj` 最大梯度 | 5.7522e-7 |
| COSMIC query 最大梯度 | 6.4473e-9 |
| `H_COSMIC_w` 最大梯度 | 5.6710e-3 |
| `proj_pre` COSMIC 列最大梯度 | 7.3302e-5 |
| FY 空 mask 对 COSMIC 输出影响 | 0 |

Smoke checkpoint：`checkpoints/run65_cosmic_fix_smoke/smoke_overfit_model.pth`  
SHA256：`281551aedc4cf907669a2e88034a3f0fd2fd9177928591314ffd94ac27bf67de`

## 四模式验收

沿用现有 P2 的 5 个时刻、1,000 个分层空间点和正式 FY/COSMIC 索引，`strict=True` 加载并冻结推理。

| 验收项 | 结果 |
|---|---:|
| FY / COSMIC 覆盖率 | 2.16% / 63.00% |
| COSMIC 覆盖点 `max|M01-M00|` | 0.8145771 |
| COSMIC 覆盖点 `max|M11-M10|` | 0.8145771 |
| 双源无覆盖点 `max|M11-M00|` | 0 |
| 重复 M00 最大差 | 0 |
| 教师冻结、严格加载、有限输出 | 通过 |

完整报告：`checkpoints/run65_cosmic_fix_smoke/four_mode/teacher_smoke_report.json`。

## 原 checkpoint 与后续决定

原 `best_fsia_model0.pth` 修复前后 SHA256 均为 `dd23cff3d1050ad7b333eef37661389f30818bf47471d872ddbe4ad8549c7fce`。

暂不启动完整重训：当前仅有 CPU 环境，而 FY/COSMIC 数据分别约 920 万/1486 万点，正式训练仍含在线局部邻域查询。Smoke checkpoint 只证明分支和验收链路已恢复，不能替代生产教师。因此代码层 P2 阶段门已恢复，但生成正式 P2/P3 标签前仍需在新 run 目录完成全量训练、常规验证及同一四模式验收。`H_COSMIC_u/v` 与 FY 对应低秩因子一样仍为零；主 `H_COSMIC_w` 已有效，不在本次最小修复中扩展 SALR-H。
