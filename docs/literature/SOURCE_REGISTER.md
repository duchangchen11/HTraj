# 来源与核查登记

日期：2026-10-14。结论只使用已明确核查的来源；本清单不是穷尽综述。

## 取得的原文

| 来源 | 核查范围 | SHA256（对应取得的 PDF） |
|---|---|---|
| [HighGraph](https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf) | full main paper, 10 pages | `9593940f1813e32723708179c5478e444a5656e1db66f8469a80423c1f73b7d4` |
| [HighGraph supplementary](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Kim_Higher-order_Relational_Reasoning_CVPR_2024_supplemental.pdf) | full supplementary material | `abef58474f3941fd3c577d2ffc8653725829bdca1a0731734a8187c497d72c80` |
| [UniEdge](https://www.hubertshum.com/publications/tcsvt2025trajectory/files/tcsvt2025trajectory.pdf) | full accepted manuscript, 14 pages; official code static inspection | `1eee06c872c021ae1ecf0fc916469b5bfe46797bc2e06d0a2d307c034402c036` |
| [GroupNet](https://openaccess.thecvf.com/content/CVPR2022/papers/Xu_GroupNet_Multiscale_Hypergraph_Neural_Networks_for_Trajectory_Prediction_With_Relational_CVPR_2022_paper.pdf) | main paper read; architecture and relational modules | `f5a74d5540c479f0e826453c276cad66587727a22dc2360ab631d6ddcc16beab` |
| [TGN](https://arxiv.org/pdf/2006.10637) | original methods section: messages, aggregation, memory, embedding | `580113858f363b80388bbf7c1b2a42482cfb9b6debdb3db2cdde8cb6db8166c0` |
| [DBGNN](https://proceedings.mlr.press/v198/qarkaxhija22a/qarkaxhija22a.pdf) | original background and architecture: time-respecting walks and higher-order graphs | `bc034fa19ddedaca1f762150f83dbd3b4c28da4740c7a82370cb8ee13843e33a` |

哈希用于版本核对；正文、补充及第三方代码临时放在 `/tmp/htraj-stage1-sources/`，不属于持久文献库，也不提交。需要重读时从来源链接重新获取并校验哈希。结构及公式定位在各分析报告中。

## 身份交叉验证

- HighGraph：CVF 出版记录与正文作者/题目/页码核对，CVPR 2024，DOI 10.1109/CVPR52733.2024.01444。
- UniEdge：arXiv 2502.02504、作者 PDF 与 [Durham 接受稿记录](https://durham-repository.worktribe.com/output/3466398/unified-spatial-temporal-edge-enhanced-graph-networks-for-pedestrian-trajectory-prediction) 交叉核查。IEEE TCSVT 35(7):7047–7060；在线 2025-02-06、卷期 2025-07。
- HoGAT： [Crossref 原始元数据](https://api.crossref.org/works/10.1080/21680566.2025.2532650) 与 [许可转载作者摘要](https://trid.trb.org/View/2643271)，摘要非出版方全文；2025-07-17 在线日期与 TRID 卷期日期分开。
- MixHop：[PMLR 原始记录](https://proceedings.mlr.press/v97/abu-el-haija19a.html)，仅核查摘要和元数据。

## 检索记录、失败和范围

检索模式：multi-source-search；原文格式：pdf-text；设计写作按 algorithmic/method/experiments、中文研究报告处理。用户要求结构分析，因此采用中文逐章精读与公式定位报告，不额外出版全文双语副本。

关键词：HighGraph CVPR 2024 pedestrian trajectory；UniEdge 2025 trajectory；HoGAT trajectory 2025；time-respecting paths graph neural networks；GroupNet multiscale hypergraph；temporal graph networks memory。来源为 CVF、作者公开稿、arXiv、PMLR、Crossref，并核对官方代码。按 title/DOI/arXiv 合并重复记录。

- 学术检索 MCP 不在会话工具中，使用网页检索定位及直接公开原文下载；未声称执行 PubMed/Scopus 搜索。
- 网页工具若干 CVF 请求失败，但 shell 直接公开 PDF 下载成功。
- HoGAT 出版方 full/abs/pdf 403，未获得正文。已取得 Crossref 元数据和许可摘要。
- GitHub 公共 API 对代码树查询返回 403 rate limit；随后合法 git clone 获取 UniEdge 固定提交，未运行其脚本。
- 搜索中曾打开 2502.08957，核对标题发现与 UniEdge 不符，已排除；正确标识是 2502.02504。
- 宿主没有 `python` 命令，统一使用 `python3`。

## 后续检索线索（尚非全文证据）

TGAT、SMEMO、EvolveHyperGraph、PCHGCN、2026 patch-wise ST graph 和 HYPA-DBGNN；应核查方向、时间 mask、阶数、path memory 与预测协议再加入结论。不能由检索摘要宣布候选首次创新。

## 术语规范

social depth=社会深度；higher-order graph=高阶图（按场景说明多跳/路径）；hypergraph=超图；time-respecting walk=时间相容游走；edge-to-edge=边间传播；best-of-20=20 样本最佳指标。TOHI-Net-A/B 是项目候选名称，均未实现。
