# Stage 1 仓库检查与发布记录

日期：2026-10-14（Asia/Shanghai）。目录 `/home/lrj/HTraj`，远程 `https://github.com/duchangchen11/HTraj.git`。

## 初始检查

| 项目 | 实际观察 |
|---|---|
| 本地目录 | 用户预建，存在，只有 `.`/`..`，无文件 |
| Git 初始化 | 初查未初始化，status/remote/branch/log 提示非仓库 |
| 未提交成果 | 目录为空；没有文件被覆盖 |
| 既有分支/远程 | 无 |
| 远程历史 | `git ls-remote` 成功退出且无输出，即无分支/tag 引用 |
| 本地/远程差异 | 双方没有既有提交，不存在历史冲突 |
| 旧项目 | 未进入、未修改、未删除 |

宿主 Git 不支持 `git init -b`，失败未产生仓库；随后使用 `git init` 与 `git symbolic-ref HEAD refs/heads/main` 安全建立 main。origin 指向用户指定仓库。提交身份根据已连接 GitHub 账户核查，只写本仓库 Git 配置，没有修改全局配置或在报告中公开账户邮箱。

## 本阶段文件范围

README、AGENTS、忽略规则、目录占位、设计 YAML、文献及方法分析、论文框架、总报告、轻量文献状态 CSV、BibTeX、来源哈希 manifest 和检查脚本。无模型训练成果。作者原文与临时第三方代码在仓库外用于核查，未复制到待提交树中。

## 发布检查协议

提交前运行 `python3 scripts/check_stage1.py`，检查必需文件、相对链接、公式/code fence、JSON、忽略规则正反例、5 MiB 文件阈值与常见密钥模式。再检查 staged 文件清单与 `git diff --cached --check`、`git diff --check`。模式检查不是完整凭据审计；同时人工检查来源与新增文本。

本阶段提交前实际检查通过：20 项必需交付、16 个目录、32 个忽略正反例；32 个非忽略文件，约 93 KB，最大文件约 8 KB；相对链接、数学/代码围栏和来源 JSON 有效，未命中常见密钥模式。`git diff --check` 通过；远程再次查询无引用，`git fetch origin` 成功且没有取得既有提交。忽略规则例外的检查采用 verbose 输出解释 `!`，避免将匹配到允许规则误判为被忽略。

阶段提交使用 `docs: establish HTraj research framework and stage1 study`。push 前再次查询远程并 fetch；若有新远程历史则停止、报告，不自动合并不相干历史。空远程时正常 `git push -u origin main`，不 force。push 后查询远程 main SHA 并与本地 HEAD 比较，检查工作树干净。

该文件是首次阶段提交的一部分，不能在提交内记录其自己的最终 SHA。**最终 SHA、push 实际结果和远程核验在用户最终反馈中记录**，也可由本地 `git log -1` 与 `git ls-remote origin refs/heads/main` 重现。若认证失败，保留本地提交；不得把流程描述当作成功证据。
