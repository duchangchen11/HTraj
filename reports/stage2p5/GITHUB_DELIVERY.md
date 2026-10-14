# Stage2.5本地交付与用户手动上传

日期：2026-10-14。项目`/home/lrj/HTraj`，origin保持`https://github.com/duchangchen11/HTraj.git`，本地分支`feat/stage2p5-social-diagnostics`。

本阶段开始核实远程Stage2 `f35b9489fc45ca4815242dd46486ce5652410219`与本地一致，main为`4b36e16540c103d7868e44ea6623f24d013aeed6`。安全从Stage2建立新分支，训练源码/协议提交`90d987799d06bef4e213952342da231e93792a41`后运行，未覆盖历史。初次普通push此源码提交时出现`fatal: could not read Username for 'https://github.com': terminal prompts disabled`，没有强推或改写远程。

**用户最新指令：不需要代理push，每次结束后用户自己在终端上传。** 因此本阶段完成实验、报告与本地提交后停止自动上传；此处不是等待认证才能继续科研的阻塞。GitHub未由本次代理上传最终成果，也不声称已在远程核实本次文件。最终交付SHA见`git log -1 --format=%H`与本阶段反馈。

已检查忽略规则和大小/敏感文件；原始数据、缓存、权重、凭据不进入提交。所有模型训练源commit与数据/subset/协议SHA保存在实际表格中；旧权重与结果保护字节检查通过，新增best/last本地保留。完整单元测试87项通过。

用户手动正常上传命令：

```bash
cd /home/lrj/HTraj
git status
git log -1 --oneline
git push -u origin feat/stage2p5-social-diagnostics
git ls-remote origin refs/heads/feat/stage2p5-social-diagnostics
```

比较ls-remote SHA与本地HEAD确认上传。若出现历史冲突/non-fast-forward，停止并检查差异，不使用force；认证失败只需处理本机认证，不丢弃本地成果。既有Stage1/2交付记录保留为当时状态，不能解释为当前远程仍空。
