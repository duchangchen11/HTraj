# HTraj research execution rules

本研究工作目录为 `/home/lrj/HTraj`，远程为 `https://github.com/duchangchen11/HTraj.git`。
论文题目：《面向复杂场景的行人轨迹预测关键技术研究》。新代码、配置、文献分析、图片、报告与轻量结果统一在本仓库管理。
不使用 `/home/lrj/PedNegoNet/` 作为本研究工作目录；不得删除此前项目。

- Stage3.5在feat/stage3p5-benchmark-audit完成协议/价值核查：606项旧文件/90个权重SHA保护；A完整8/12历史上下文与B原partial-mask分开，五个录像隔离候选仅统计，只students003原val70区域做seed17四个现有神经基线run各30epoch，原val未混训练、ETH封存。A GNN弱于PedBase，B仅小收益，无稳定独立社会优势；148测试通过，公开Social-STGCNN静态核查未复现，recording隔离不等新独立确认。不得自动新增复杂模型/高阶、开发B架构/第五章、评价ETH或正式五折；只本地commit、用户手动push，完成后停止，后续需新指令。
- 事实写明原始来源和定位；推断与创新假设标注“待验证”。不得虚构实验结果或把已有方法改名作为创新。
- 保留已有文件和未提交工作。同步前检查 status、remote、branch 和历史。
- 禁止强制推送、删除历史、`git reset --hard`、覆盖未提交成果。历史冲突时停止并报告。
- 每个研究阶段结束必须检查忽略规则、敏感文件、文件大小和 `git diff --check`，完成本地提交。用户在Stage2.5明确改为自己从终端上传：今后不自动push，交付最终SHA及正常push命令；如用户再次明确授权上传，再按当次指令执行。
- 数据、缓存、权重和凭据不提交；源码、配置、分析报告、轻量表格和完成图片可以提交。
