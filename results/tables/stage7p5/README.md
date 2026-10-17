# Stage7.5真实分析数据

只使用Stage7.4 execution_v2 validation NPZ；没有新增正式推理/训练，也不读取test误差。官方源码、权重及原始文件SHA见input_and_code_provenance.json及configs/stage7p5/protocol.json。

- `hotel_targets.csv.gz`, `zara2_targets.csv.gz`：每个来源录像/末观测帧/行人ID一行；历史分组、社会描述、三种CV、各候选口径运动指标、K前缀覆盖。没有原始轨迹坐标。
- `*_heads.csv.gz`：每个目标20行；逐head的ADE/FDE、速度、方向、转向、尺度归一化结果及赢家平分贡献。完整精度用于复核，属于派生指标，不是模型权重或原始数据集。
- `target_provenance.csv.gz`：两阶段目标/历史/GT SHA映射。Stage7.4数组SHA包含dtype+shape+bytes；本阶段目标表同名SHA是bare float32 bytes。数据本身逐目标一致，不能直接比较两种编码的哈希字符串。
- `complete_group_statistics.json`：**最终组统计**，窗口/行人等权及每指标有效样本数。`source_by_speed_statistics.json`保存来源×运动组；缺组n=0且没有虚构指标。
- `group_statistics.json`：保留的初版结果，可能漏掉首行未定义的方向指标；被complete版本替代，不作为正式方向统计。
- `person_equal_metrics.csv.gz`, `source_macro_statistics.json`：录像内ID等权、来源录像宏平均，和窗口平均不同。HOTEL/ZARA2权重fold平均不是独立确认。
- `K_subset_curves.csv.gz`：预先固定的21套嵌套head顺序（原顺序+20套随机），各K、来源和历史组；所有min指标都是GT oracle。
- `head_contributions.json`, `head_imbalance_summary.csv`：按来源/组/权重口径的head贡献、熵有效head数。频率不用于部署排序。
- `timestep_errors.csv.gz`：每未来步误差；使用整条轨迹的固定赢家，不逐步换head。
- `motion_speed_transitions.csv.gz`：真实/预测未来平均速度相对历史平均速度的变化，全部为事后诊断。
- `history_error_associations.csv`：行人ID均值之间Spearman相关，无窗口IID p值、无因果主张。
- `person_cluster_intervals.json`：200次来源内行人cluster重采样的**行人等权均值区间**；不要配给窗口均值，不能当成新录像泛化区间。行人之间仍可能相关。
- `scale_adjusted_regressions.json`：来源固定效应、行人加权的描述性尺度回归；高速类别来源支持稀少，不作因果解释。
- `stage7p4_target_metric_equivalence.json`, `verified_prediction_inputs.csv.gz`, `history_preservation.json`：目标/GT/指标对齐、1278个预测文件、旧成果字节保护。
- `literature_comparison_matrix.*`, `literature_sources.json`：论文/代码版本、阅读边界、下载/访问失败。论文PDF与第三方源码只存本地忽略缓存，不提交。
- `unit_tests_final.*`：最终433项测试全部通过；初版432项结果保留。
- `failed_attempts.json`, `aggregation_failed_attempt.json`, `nullable_export_failed_attempt.json`, `aggregation_repair.json`及一致性检查中的首次SHA编码失败：真实失败和修复，不能省略后宣称首次全通过。失败导出现场存忽略缓存；既有Stage1—7.4未改。

误差单位米，速度m/s，角度rad；归一化误差无量纲。空值代表不适用/方向不可靠，绝不填0。历史速度阈值0.05/0.4/1.8m/s；方向可靠性0.05m/s和0.24m净位移；历史转向组阈值30°。所有定义见冻结协议。

执行入口使用隔离环境`experiments/stage4/venv/bin/python`：

```bash
python scripts/stage7p5_prepare.py
# 在Git冻结协议和源码后，分析；默认独占新输出，不覆盖既存结果。
python scripts/stage7p5_analyze.py
python scripts/stage7p5_complete_statistics.py
python scripts/stage7p5_check_outputs.py
python scripts/stage7p5_verify.py
python scripts/stage7p5_reports.py
```

这些命令为流程说明，已有输出目录不能直接重复执行；应使用新输出位置并保留原结果。prepare依赖Stage7.4本地忽略缓存；下载PDF和旧checkpoint不在Git里，单独只有仓库不能伪装成已取得这些资源。执行源码commit与补充脚本SHA已记录，最终Git commit提供完整交付版本。
