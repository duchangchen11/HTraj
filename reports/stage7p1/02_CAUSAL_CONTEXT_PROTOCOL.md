# 严格历史上下文协议与冻结目标

官方版本 `e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5`；[源码与配置](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。本阶段模型执行代码版本 `6d07b7f4c697aa79142f383fe4040d133e1531b4`，逐文件SHA见 `results/tables/stage7p1/execution_setup.json`。

**A：HTraj-AgentFormer-A_offline_complete20-v1**仅为离线完整20帧参照，完整未来存在性可以决定其场景节点。本阶段不运行A评价，也不以A替代B。

**B：HTraj-AgentFormer-Causal8x12-v1**是本轮主准入：8帧历史、12帧未来监督、0.4秒、米制。原始四列来源由 `Recording.read`重新解析；上下文 `C=intersection(ids(X_t),t=0..7)`，按ID排序，不设人数或距离上限。不完整历史节点记录并排除，不填补未知位置/速度。这个历史规则限制了可用信息，绝不称为全部历史可见行人均已纳入。

上下文构建只调用历史8帧的字典查询；确定上下文后，标签单独读取。监督集合S严格匹配Stage5.1冻结target CSV；S的历史与12帧GT逐目标SHA均一致。上下文C不读取S或邻居未来存在性。未来缺失坐标在标签审计中为NaN，不能当零坐标标签。

| 留出录像 | 划分 | 支持窗口/有监督窗口 | 冻结目标窗口 | 混合标签窗口 | 最大历史完整人数 |
| --- | --- | --- | --- | --- | --- |
| biwi_hotel | train | 2221/2121 | 27145 | 1818 | 73 |
| biwi_hotel | validation | 149/94 | 318 | 94 | 15 |
| crowds_zara02 | train | 1959/1684 | 23545 | 1440 | 73 |
| crowds_zara02 | validation | 172/172 | 1092 | 123 | 13 |

“混合标签窗口”只计有监督目标且存在未来不完整的上下文节点。HOTEL支持区共149窗口，其中94窗口含冻结目标；这94个全部混合标签。其余窗口仍保留上下文统计，未制造额外监督目标。ZARA2的172窗口中123个混合标签。目标窗口不是独立行人数。

## 数据版本与支持区域

HOTEL manifest SHA256：`e3f444dd718d0c358b817d5350ebd29e8b91c302beb153929c0302a0ebeae9a1`。

ZARA2 manifest SHA256：`9ed3d7975b6288ec254de7152edbb166aa9d53f4fcbac9e51de43ada0bf13246`。

五个非ETH原始文件SHA、四份target CSV SHA、完整人数分布见 `data_audit.json`；逐窗口上下文ID、完整未来目标ID、逐人未来有效长度及部分历史排除ID见 `context_windows.csv.gz`。训练与留出录像隔离，旧validation未加入训练，20帧支持区域不改变。ETH轨迹及指标均未读取。两开发录像已被历史实验曝光，不是独立确认集。

## 工程覆盖补充

真实测试需要含监督目标的N1/N2及全标签多人场景。冻结验证支持不足时，仅从另一fold已冻结的原train区域取首个对应人数的窗口作为功能测试，0更新；不改变训练/验证划分。规则在执行前以 `real_case_coverage_addendum.json`和提交 `6d07b7f`冻结。该文件“no causal N1”的准确含义是**无含监督目标的验证N1窗口**；HOTEL仍有14个无监督目标的历史N1窗口，已保留在全支持统计中。

## 模型语义边界

没有新增监督mask、KL mask、未来token mask或输出节点类型。`original_training_packet`遇到C中任意未来缺失即拒绝。推理接口的未来NaN/零有效mask只是满足原 `set_data`结构，完全不进入infer计算，不是可用训练标签。原始loss只在C全部有真实12帧GT时执行，未改loss数学含义。
