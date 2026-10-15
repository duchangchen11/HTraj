# Stage4：公开基线核查与非ETH checkpoint验证

## 阶段结论

已建立**可核验、可运行的Social-STGCNN官方维护版本推理基线**。四个非ETH官方完整测试分区实跑成功；EqMotion完成源码、协议、环境与权重审查，但官方权重不完全匹配当前代码，性能验证已停止。没有增加新模型模块、没有训练、没有读取ETH评价成绩，也没有自动推送。

推荐下一阶段以Social-STGCNN维护版作为工程起点，先做统一协议及小预算从头训练核查。EqMotion暂不作为正式主干，HTraj原Dynamic-GNN保留为历史对照。目前不足以启动正式新方法设计或宣称新论文优势。

## 版本和保护

- 基础分支 `feat/stage3p8-motion-reliability`，基础提交 `f12146143514d2b6eb6cf14877cb5b4cae454543`。
- 本阶段分支 `feat/stage4-public-baseline`；运行前协议提交 `3736d0e`，补充检查提交 `26075a24ed646505ae5e8a269ee86ab5f4c6b697`。最终交付提交以本分支最后Git commit为准；结果记录执行时commit及脚本SHA，避免伪造尚不存在的最终SHA。
- `configs/stage4/protected_artifacts.json`保存1027个原跟踪文件及142个旧checkpoint，共1169项SHA256；交付检查逐项验证。Stage1–3.8源码、配置、结果、报告及旧权重没有覆盖或删除。
- Social官方冻结 `333d3a57b4d2705e129b21aefefa09c79b2b9ae1`；EqMotion冻结 `5aec2e0b61c511fa93a24138dd90da59a089084b`。独立路径 `third_party/stage4/`，MIT许可证保留，上游源文件字节未改。
- 独立Python环境 `experiments/stage4/venv`，没有导入旧项目自定义模型/缓存。数据、checkpoint、环境和长日志只在被忽略目录；Git仅保存代码、报告、配置和轻量误差/日志摘要。

## 真实Social-STGCNN结果

各数值为固定权重下采样seed17/42/2026均值，单位米；不是独立训练三种子。每次完整评价2771窗口、33473目标窗口；UNIV包含两个原始录像。

| 场景 | 目标窗口数 | K1高斯均值 ADE/FDE | K20独立最优 ADE/FDE |
| --- | ---: | --- | --- |
| HOTEL | 1053 | 0.657375 / 1.321567 | 0.407989 / 0.676115 |
| UNIV | 24334 | 0.684362 / 1.356624 | 0.488905 / 0.912310 |
| ZARA1 | 2253 | 0.506405 / 1.050879 | 0.330804 / 0.517729 |
| ZARA2 | 5833 | 0.444911 / 0.896088 | 0.303870 / 0.481329 |
| 四场景等权宏平均 | — | 0.573263 / 1.156289 | 0.382892 / 0.646871 |
| 目标窗口加权 | 33473 | 0.629808 / 1.254689 | 0.443474 / 0.803219 |

另存K1首采样（宏平均0.689692/1.298289）、每种子结果及301257行逐目标误差（gzip约6.7MB）。K20 ADE/FDE分别取最小，可能来自不同样本，不能与K1或此前HTraj成绩混排。12次上游评价函数输出与独立重算均一致到1e-6以内。7563参数；loader与评价核心计时合计约9.5分钟，RTX3080实际执行。

## 必须保留的失败和限制

1. **官方维护版不等于精确CVPR2020版。** README给出的论文commit为 `ebd57aa...`；当前版图卷积输出通道和图乘法已改变。本轮没有改上游网络，但也没有运行精确2020架构、重新训练或复现论文全表。
2. **EqMotion没有性能结果。** 计划HOTEL/ZARA1各最多128例，实际权重检查时停止，评价0目标。HOTEL多余399键；ZARA1缺失184且多余399键。未改类别数、未随机补权重、未把strict=False当成复现成功。
3. EqMotion随机权重合成结构检查通过，最大旋转一致性误差约3.08e-5米；这只能说明源码几何性质可运行，不能代替官方checkpoint精度。官方HOTEL/ZARA2预处理每样本仅1人，不能用其成绩证明社会交互。
4. Social原始五录像文件与HTraj SHA一致，但官方test整录像、完整20帧上下文与HTraj原val时间区域/历史上下文不同。接受窗口已复核没有非10帧间隔，名义0.4秒；仍不是HTraj协议完全相同的证据。
5. 五个非ETH录像已曝光，不是独立确认集。本轮ETH不评价、不读成绩；但官方非ETH预训练模型可能在上游见过ETH训练数据，未来ETH最终测试必须使用重新建立的排除ETH训练流程。
6. 没有统一训练、无交互消融、容量匹配或新方法性能，因此不证明图结构贡献、不比较HTraj与公开模型高低、不宣布投稿条件已满足。

## 测试、日志与执行证据

训练前208项测试通过；补充权重不匹配审计及EqMotion合成等变检查后，完整210项测试通过（8.67秒），`final_tests.xml`为真实结果。交付检查记录历史保护、上游源码/资产SHA、参数量、逐目标误差汇总、K20与K1合法关系、非ETH白名单、原始时间支持、敏感/大型文件和Git whitespace检查。

网络获取曾出现Git TLS中断，限定重试后使用按Git blob校验的raw HTTPS补齐；没有将失败下载当作可用资产。文献技能的OpenAlex fallback发生SSL错误，最终用CVF/arXiv/作者官方仓库核实来源；不称穷尽检索。早期Git带大小的元信息查询触发惰性对象获取，已停止，最终只按文件名清单取白名单资产；没有签出、解析或评价ETH轨迹/成绩。详细事实保存在 `setup_provenance.json`、`execution_events.json` 及本地原日志。

## 交付索引

- `reports/stage4/01_SOCIAL_STGCNN_AUDIT.md`：论文身份、原版差异、网络公式、图定义、许可和环境。
- `reports/stage4/02_SOCIAL_STGCNN_REPRODUCTION.md`：完整真实非ETH结果、K口径、耗时及复现边界。
- `reports/stage4/03_EQMOTION_AUDIT.md`：等变性、交互、数据筛选、权重失败及合成检查。
- `reports/stage4/04_PROTOCOL_COMPARISON.md`：三实现协议和公平性限制。
- `reports/stage4/05_BASELINE_SELECTION.md`：十个研究决策问题、下一步门槛、已有研究与时间估算。
- `results/tables/stage4/`：原始36行Social结果、12行三种子汇总、18行聚合、逐目标gzip、来源/环境/配置hash、Eq失败诊断、测试和交付检查。
- `configs/stage4/`：预注册协议、官方文件/资产SHA、历史保护清单。
- `scripts/evaluate_stage4_social.py`、`evaluate_stage4_eqmotion.py`、`audit_stage4_eqmotion.py`、`audit_stage4_protocol.py`、`summarize_stage4_results.py`、`verify_stage4_delivery.py`：执行及核验入口。失败的Eq入口保留，不要求重新运行或放宽检查。

本阶段完成本地提交后停止。用户自行执行：

```bash
git push -u origin feat/stage4-public-baseline
```

下一阶段建议是协议对齐及有限公开基线训练核查，预计工程准备1–2个工作日；正式训练耗时未测，须先固定小预算吞吐试验再估算。本次不继续训练、不设计新网络、不执行ETH测试。
