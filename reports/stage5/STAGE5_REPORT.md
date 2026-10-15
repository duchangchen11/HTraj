# Stage 5：公开基线选择总报告

**结束TOHI与当前PTN G0/G1家族的结构迭代。主候选选择SocialVAE，备用选择SGCN的完整历史发布；两者均附适用条件。EqMotion暂不可用，HighGraph暂未定位公开实现。当前尚不具备直接宣布或开发论文新模块的证据。**

项目`/home/lrj/HTraj`，分支`feat/stage5-baseline-selection`，基础`e6e9700eba015140d00e989a2aaaf7d7a88bd00d`。2,385项旧源码、报告、配置、实验文件及checkpoint的SHA256已封存，最终核验见`delivery_checks.json`。没有改写旧指标、删除文件、覆盖旧checkpoint或使用其他项目模型。官方源码/权重在本地忽略目录独立管理。

## 主要发现与研究决定

|对象|证据等级|真实发现|决定|
|---|---|---|---|
|EqMotion|完整可达历史+实际strict检查|28个提交，ETH/UCY只有一套代码；HOTEL多399键，ZARA1缺184且多399键；没有匹配版|停止本轮调试，暂不可用，不执行预测、不填ADE/FDE|
|HighGraph|正式论文+官方页面/仓库定向检索|发表可核实；未定位可冻结官方源码，是宿主插件|保留文献参考，不能列为已可复现|
|SocialVAE|官方源码/权重+实际有限推理和训练smoke|两个权重严格匹配，2.144M参数；256目标推理、1步从头训练和严格恢复通过；loader存在原始数据未来边界依赖|有条件主候选，先修订数据与选模契约，再验证基线|
|SGCN|官方历史源码/权重元数据|同一历史提交有完整代码和权重，当前main已删除代码；节点轴卷积仍有结构风险|有条件备用；尚未加载、运行或公平评价|

SocialVAE冻结`eb23cbae562bac2c2dcf0d81529e6ab2ac280962`；SGCN备用完整发布`5153702d8ae32a2b0586b093d4e094aee08b0f9b`。EqMotion冻结`5aec2e0b61c511fa93a24138dd90da59a089084b`；HighGraph没有找到commit，不填假版本。

## 实际执行的有限指标

SocialVAE官方离线loader各录像预定前128目标窗口，seed17、batch16、无FPC。单位米：

|录像|K1 latent-prior均值 ADE/FDE|K1首样本 ADE/FDE|minADE/minFDE@20|
|---|---:|---:|---:|
|HOTEL|0.185164 / 0.358718|0.286168 / 0.589197|0.094055 / 0.131151|
|ZARA2|0.466260 / 1.029628|0.637564 / 1.394796|0.249112 / 0.447714|

这些是**存在已记录协议风险的功能诊断值**，不是论文全录像复现、严格历史协议成绩或与旧HTraj公平比较。两个K20最小值分别选择，不要求来自同条候选；K1 latent均值不等于轨迹分布期望。重复窗口只有9/13名不同行人，不当作256个独立样本。没有读取ETH测试成绩，也没有运行ETH预测。

最重要的反例：仅改变一个末帧新进入邻居的未来，保持原始8帧观测和目标未来不变，官方loader补入的历史速度变化0.8米/步，同一权重预测平均变化0.487453米。forward截断未来张量仍不能排除预处理泄漏。固定真实子集中末帧新近邻标记为1/128与0/128，不外推全局比例。本阶段记录问题，没有偷偷修loader后继续冒称原版复现。

## 第四章与小论文

既有证据不支持当前高阶、历史关系GRU、简单消息新鲜度或PTN图家族的稳定独立收益。原PedBase本已相对运动编码；普通坐标归一化、CV残差、图稀疏、社会注意力和等变性均非可直接声明的新创新。这些负结果只约束已测原型，不等于社会信息普遍无价值。

下一问题应是：**严格限定历史信息、固定强个体运动基线与采样预算后，成熟模型的社会分支是否具有可归因的预测增益；这种增益是否依赖邻居运动可观测性？** 先建立SocialVAE严格无泄漏的数据与开发选模流程，再做从头训练的Full/完整NoSocial/近容量控制及合法关系负对照。必须覆盖初始社会隐状态和逐帧注意力两条通路，不仅关闭一个图/attention开关。

这条路线目前最接近可执行的后续研究，但还没有支持新方法小论文的性能和新颖性证据。若严格基线依然不能超过配对CV和容量控制，就停止社会模块扩展，重新审查任务/数据集；不自动增加高阶、关系GRU或门控。

## 文件、测试与交付

报告：

- `reports/stage5/01_PUBLIC_MODEL_SHORTLIST.md`
- `reports/stage5/02_EQMOTION_VERSION_AUDIT.md`
- `reports/stage5/03_REPRODUCTION_FEASIBILITY.md`
- `reports/stage5/04_MODEL_SELECTION.md`
- `reports/stage5/05_CHAPTER4_RESEARCH_PLAN.md`
- `reports/stage5/STAGE5_REPORT.md`

`results/tables/stage5/`保存官方版本、完整严格报错、类别元数据、数据/权重SHA、源文件清单、固定配置、逐目标误差、原始日志、单步梯度/耗时/显存、255项测试结果和交付检查。全部255项测试通过，其中边界反例测试成功检出了loader问题，不能解读为部署因果性通过。源码零修改；只增加审查/包装器、测试和报告。完整训练、五场景实验、ETH测试、新网络均未启动；仅有预注册的一个优化步用于训练功能检查。

预注册commit为`daaa5e5b0d00254bff1560fb92fccb36aa787028`，最终交付SHA以`git log -1`为准。仅完成本地commit，不自动push。用户自行执行：

```bash
git push -u origin feat/stage5-baseline-selection
```

出处：[SocialVAE作者项目](https://motion-lab.github.io/SocialVAE/)、[官方源码](https://github.com/xupei0610/SocialVAE/tree/eb23cbae562bac2c2dcf0d81529e6ab2ac280962)、[EqMotion官方源码](https://github.com/MediaBrain-SJTU/EqMotion/tree/5aec2e0b61c511fa93a24138dd90da59a089084b)、[SGCN历史完整发布](https://github.com/shuaishiliu/SGCN/tree/5153702d8ae32a2b0586b093d4e094aee08b0f9b)、01号报告的HighGraph正式发表记录及官方搜索来源。未运行实验均明确留空或标记未验证，不用0替代。完成本地交付后停止，等待下一阶段指令。
