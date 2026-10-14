# Stage 2 GitHub 交付阻塞记录

日期：2026-10-14。目录 `/home/lrj/HTraj`；origin 为 `https://github.com/duchangchen11/HTraj.git`。

Stage 2 开始前，本地 main 与远程 main 均为 `4b36e16540c103d7868e44ea6623f24d013aeed6`。安全创建研究分支 `feat/stage2-tohi-a`，未修改 main、旧项目或旧历史。

已保留的本地研究提交：

- `c2a04752ac2a5d3e713c060ccdcde49dab9f69c2`：源码、数据协议、配置、coverage与训练前测试。真实 smoke 运行引用此提交。
- `143bae5`：真实训练结果、状态诊断、最终JUnit与报告整理。
- 随后的交付记录提交：记录本次认证阻塞，最终SHA由 `git log -1` 及用户最终反馈给出。

提交前检查：59项测试通过；五个实际fold时间支持隔离，raw哈希匹配；四模型的manifest/config/seed/budget一致，训练loss下降和ADE/FDE有限。所有可提交文件合计约4.4MB，最大单文件约694KB（划分manifest）；没有原始数据、checkpoint、预处理缓存、环境或匹配到的常见密钥模式。`git diff --check` 与 staged检查通过。

再次 fetch/查询：远程仍只有原Stage1 main，没有同名研究分支；不存在本地/远程历史冲突。

执行正常推送：

```bash
GIT_TERMINAL_PROMPT=0 git push -u origin feat/stage2-tohi-a
```

退出码128：

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

因此 **GitHub Stage 2 发布未完成**，不能声称远程已看到本阶段文件。只读GitHub访问/连接不等于本机Git凭据可用；已连接工具的commit接口也无法保留本地作者/时间元数据，故没有用它建立与本地不一致的新历史。

未强推、未重写提交、未改凭据、未启动完整训练。下一步仅需配置本机有效认证后检查远程，再正常推送本分支；若远程同名分支出现不同历史，先停止报告。研究材料无需重新生成，真实结果无需重新训练。
