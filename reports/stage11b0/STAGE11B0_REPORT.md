# Stage11B-0总报告

**工程训练GO；正式基线多轮训练仍未获批准；SCI新算法NO-GO。** 用户授权只覆盖本轮≤3600秒工程测量，完成后停止。

1. 从Stage11A+已推送fba0e46建立feat/stage11b0-ynet-training-admission；训练前代码配置冻结`6166eb7e220ea2fbdb9394b50e1ea7714057d504`。官方源`a345ada1557f53dc2acb4c07f71d340d366980da`不变，继承隔离venv，没有修改Stage1—11A+。
2. 数据身份审核通过，train7,496/25、val998/5、test2,829/17不变；测试pkl不反序列化。分割语义/生成来源限制继续单列，不冒充完整协议冻结。
3. 官方补充材料新证据确认五stuff+dummy及作者所述train-only分割预训练；SDD编号/ignore和当前checkpoint成员仍未核实，原训练池新val可能有语义曝光。
4. 新adapter沿用原Y-Net随机核心、goal/path损失、GT训练teacher forcing、Adam1e-4/seed17；冻结分割器eval，参数及BN不变；明确目标batch4与scene1，无26scene cap。
5. CPU前置12项通过；首轮临时checkpoint相对路径错误已修正，原始7个setup错误完整gzip保留，未跳过。全仓库544项最终通过，日志见full_unit_tests.xml。
6. 真实smoke3步，完整遍历1882步/7496目标，恢复探针2步，总真实optimizer调用1887。完整遍历从与smoke相同随机初态重置开始，单列重复访问，不把探针当新epoch。
7. 整轮实测453.825s；参数更新、分割冻结、无GT推理及Adam/RNG恢复后的同一步一致性通过。没有val/test成绩、没有训练ADE/FDE、没有收敛声明。
8. allocated峰2.457GiB/reserved 4.582GiB，本次25场景target4均无OOM；不外推为任何batch/增强必可运行。
9. 30/50/100/300完整遍历预算见04与budget_estimate.json。八倍增强访问59,968、200变体的GPU耗时只是由真实训练外推，未在本轮运行；正式验证与额外存储开销未测。
10. 工程GO的证据矩阵A–J见go_no_go_matrix.csv。正式基线计划需审核开发划分、语义曝光声明、选模规则和足额预算。若坚持未知场景语义，需另行无评测标签曝光的分割方案。不能将工程成功称创新。

建议下一阶段先审查上述正式方案和八倍增强训练成本，再决定是否批准有限正式复现；当前不自动启动Stage11B、多seed或300轮。完整数据遍历证明路径可训练，不证明预测能力或论文新颖性。

历史SHA、真实资源/权重/config与最终测试证据在results/tables/stage11b0。最终提交SHA以Git交付为准；原checkpoint与历史结果不覆盖。
