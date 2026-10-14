# Stage3.6总报告：强个体与冻结社会残差归因

基础2abca51dd42489a06668e6430531b984dff14641；feat/stage3p6-frozen-ego；训练前源db2ea1c2eedcecbb08c192037bbe7a0985755edf。740项旧成果/98个旧checkpoint保护，无旧源码/配置/报告/结果修改。

实际完成A协议两录像，seed列表[17]；16神经run×30epoch，共480epoch，训练3212.02秒/53.53分钟，2820条best/last逐目标配对记录；CV和Ego-Only解析另列。无高阶/关系GRU/新loss/B/ETH或旧项目模型缓存。

| 录像 | 模型 | schedule | best轮 | best ADE/FDE | last30 ADE/FDE | 参数 | 训练秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crowds_zara02 | CV | 解析 | — | 0.296306/0.667523 | 不训练 | 0 | — |
| crowds_zara02 | PedBase | constant | 29 | 0.282905/0.630561 | 0.298853/0.659355 | 35352 | 193.31 |
| crowds_zara02 | PedBase | step10_20 | 22 | 0.276696/0.619606 | 0.284282/0.631653 | 35352 | 194.15 |
| crowds_zara02 | CV-Residual | constant | 24 | 0.276902/0.617161 | 0.285986/0.639910 | 35352 | 210.81 |
| crowds_zara02 | CV-Residual | step10_20 | 8 | 0.277524/0.622087 | 0.278843/0.622135 | 35352 | 208.31 |
| biwi_hotel | CV | 解析 | — | 0.329516/0.611038 | 不训练 | 0 | — |
| biwi_hotel | PedBase | constant | 24 | 0.503319/1.005511 | 0.526877/1.052827 | 35352 | 242.11 |
| biwi_hotel | PedBase | step10_20 | 27 | 0.516671/1.017872 | 0.521228/1.026156 | 35352 | 238.72 |
| biwi_hotel | CV-Residual | constant | 1 | 0.401317/0.792782 | 0.742461/1.547134 | 35352 | 265.10 |
| biwi_hotel | CV-Residual | step10_20 | 1 | 0.401317/0.792782 | 0.594714/1.236998 | 35352 | 263.12 |

## 冻结信息来源结果

| 录像 | seed | 模型 | best轮 | best ADE/FDE | last30 ADE/FDE | 新增参数 | 训练秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| crowds_zara02 | 17 | Ego-Only | 解析 | 0.276696/0.619606 | 不训练 | 0 | — |
| biwi_hotel | 17 | Ego-Only | 解析 | 0.329516/0.611038 | 不训练 | 0 | — |
| biwi_hotel | 17 | Ego-Residual | 1 | 0.401178/0.803155 | 0.514570/1.064949 | 14425 | 144.74 |
| biwi_hotel | 17 | Social-Current | 1 | 0.390293/0.771402 | 0.604356/1.297530 | 14425 | 200.41 |
| biwi_hotel | 17 | Social-Previous | 1 | 0.389374/0.768701 | 0.550909/1.170979 | 14425 | 198.46 |
| biwi_hotel | 17 | Social-Random | 1 | 0.390286/0.770449 | 0.581645/1.262894 | 14425 | 234.26 |
| crowds_zara02 | 17 | Ego-Residual | 2 | 0.276356/0.619741 | 0.277168/0.620995 | 14425 | 116.77 |
| crowds_zara02 | 17 | Social-Current | 2 | 0.277515/0.620893 | 0.317651/0.712846 | 14425 | 156.68 |
| crowds_zara02 | 17 | Social-Previous | 2 | 0.277097/0.620228 | 0.323164/0.729582 | 14425 | 157.41 |
| crowds_zara02 | 17 | Social-Random | 2 | 0.277339/0.620522 | 0.319635/0.714126 | 14425 | 187.66 |

## 判断

社会模型未在两个录像同时达到相对anchor及同容量控制的预定收益，按规则不运行42/2026、不堆叠复杂模块。冻结排除了本轮基础ego被社会训练更新的因素；仍缺同结构联合训练反事实，无法给过去失败作唯一因果归因。Current/Previous差值、Random实际强度及完整分组如实报告。

| 录像 | Current−Previous ADE/FDE(m) |
| --- | --- |
| crowds_zara02 | 0.000418/0.000665 |
| biwi_hotel | 0.000919/0.002702 |

## 交付与复核

七报告在reports/stage3p6；protocol/保护清单在configs/stage3p6；results/tables/stage3p6保存完整配置/source/raw/manifest/protocol/hash、真实每epoch、best/last逐目标、分组、状态/门控/梯度/随机图、anchor SHA及预设停止判定。165项训练前测试通过，最终unit_tests.xml与check_stage3p6.py另核查；未运行实验不填0。权重在experiments/stage3p6、git忽略，本地保留；不上传数据/缓存/权重。

下一步先依据同容量差值与跨录像方向评估可靠运动主路径，必要时另授权同结构联合训练反事实或条件复核；本阶段不继续开发。用户自行git push -u origin feat/stage3p6-frozen-ego。

## 最终核查与核心结论

166项完整测试通过；740项旧文件及98个旧checkpoint SHA未变；16个真实run/480epoch/114480次optimizer更新，32个新best/last权重均本地忽略。全部社会anchor数值参数前后SHA一致，47个无合法消息目标在三社会变体best/last逐目标误差保持anchor、修正/gate精确0。全部组与空组保存，没有ETH指标。

HOTEL CVR last训练ADE0.401987优于训练CV0.485746，验证0.742461却弱于验证CV0.329516；无社会模块也有泛化失败。冻结后社会仍整体退化，支持先解决运动基线/录像泛化，不能简单归因给本轮ego更新。旧联合GNN的严格因果问题仍需同结构反事实。Current在两录像均弱于Previous，无新鲜度优势；小交叉组收益与弱随机图证据不足以确定社会正式网络。按预设停止规则不运行42/2026，不继续堆叠模块。
