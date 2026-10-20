# 三篇近期工作的查重缺口更新

使用nature-academic-search的multi-source-search流程；未挂载学术MCP，CrossRef→OpenAlex→Semantic Scholar以及正式出版页/作者代码有限回退。9次metadata响应在literature_access_update.json及cache，网络失败/非JSON不包装成成功全文。搜索范围仅三篇指定文献，不换新模型。访问日期2026-10-20。

|论文|本轮实际增加的证据|仍未核实|
|---|---|---|
|[Completed Interaction Networks](https://doi.org/10.1109/TMM.2025.3542967)，TMM2025|核查正式元数据/摘要；审查自称作者实现的[CINet-main](https://github.com/Jayzhou3219/CINet-main)核心CINetr、CINetv、another_goal与STTL源码|完整论文数学方法未取得；repo README自称官方，独立出版方→源码归属未交叉确认|
|[McTrans](https://doi.org/10.1016/j.jvcir.2026.104723)，JVCIR2026|出版页引言/摘要/可见方法概述，CrossRef与Elsevier官方TDM响应|TDM200只含coredata、无方法正文；未找到经核实作者预印本/公开代码|
|[GDS-TGTF](https://doi.org/10.1016/j.eswa.2026.134731)，ESWA2026-10-14 pre-proof|出版摘要/亮点与官方TDM元数据|无完整公式、训练细节或可核实公开代码，不能据摘要断言无语义可靠性机制|

CINet源码冻结`3ef2ee76970cd6110e46284f70532bd60d7df4fe`，49个小源码/README文件blob校验，静态审查未执行、未复制入Git。可见路径为6通道环境图+轨迹热图预测goal概率，结合时空分支与目标距离、当前时间等，loss含轨迹MSE与goal BCE。已覆盖场景/社会/短期goal联合建模。已读文件未能确认有校准的语义不确定性目标，但这**不是整篇论文没有该机制的证据**。可见代码还有缺失Ablation_Study导入、hard-coded cuda:1、forward内构造模块等审查风险；未运行，不能将其当作验证过的官方算法。

McTrans公开引言说明learnable goal token、环境CNN、层级cross-attention和逐步增强目标损失；这些不是待设计的新贡献。GDS公开摘要说明结构/运动双流、融合和目标逐级refinement。精确损失/可靠性权重/动态路径修正是否与拟议机制重合仍未知。

网络记录：GitHub API403限流后用git tree；首次代码过滤过宽下载到少量TXT数据，未解析、未预测/评价，随后明确排除data/output，仅小源码审查；中途异常和初始日志保留。Elsevier TDM返回HTTP200并不意味着全文可用，实际1840/1875字节只含coredata。没有使用第三方求助站/未经授权全文，没有绕过出版方访问限制，也未无限等待。

Stage11A已有Y-Net/SSU/UATP等重合结论保持不变。当前三篇全文缺口**没有全部关闭**。不能证明语义不确定性、目标候选可靠性尚未有人处理；不批准SCI新算法开发。本阶段没有提出新网络。
