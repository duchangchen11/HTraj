# 无语义训练工程与预算

**正式训练未启动。完整epoch=0，正式optimizer更新=0。**

预检75个基准batch+2次严格恢复重放更新，均属于临时模型；不是完整epoch或性能实验。

新输入14通道=6全零语义+8原始历史距离热图。语义float32全画布含padding为零；
原Goal与Path decoder、softargmax、损失及GT waypoint训练teacher forcing保持。
验证接口只接收冻结history-only缓存，不接收GT；定义为先保存998条K20结果并校验完整性，再读取GT评分。
K20仍为原CPU multinomial每sample_id独立SHA种子、10000有放回proposal、原19中心KMeans与1个softargmax。
模型选择仍为完整验证中窗口加权minADE20最低，完全相同则较早epoch；配对FDE同一checkpoint。
科研主比较固定Epoch30对Epoch30，五录像等权macro；best比较单列次要分析，不能事后挑有利方案。
科研门槛分母统一NoSemantic：macro ADE改善≥2%，macro FDE语义退化≤1%，至少3/5录像ADE支持语义，
OTHER4合计语义ADE退化≤1%。最后一项把用户“明显反向”事前具体化为1%，未依据验证误差选择。
Delta_ADE/FDE=NoSemantic−Semantic；Delta_Cover=Semantic−NoSemantic，正值均支持语义。


实测GPU相关父进程累计 **66.674948s**；硬上限18000s，安全阈值16800s。

|成本|秒|性质|
|---|---:|---|
|临时模型预检完整进程|66.674948|实测，包含启动、构造、更新、K20、保存和退出|
|按25场景固定batch投影单轮|482.656582|投影，未执行完整epoch|
|K20单目标，六次调用平均|5.285585|两训练目标，未评价验证GT|
|K20单目标最大值|5.358410|同上|
|Checkpoint保存并校验|0.721301|实测临时模型|
|保守全计划|56627.592801|30轮及7次验证投影，非已发生耗时|
|仅第一次验证较慢，其余沿用旧验证上界|23919.672562|情景估算，仍超过硬上限|
|全部沿用原语义模型历史耗时|16388.867259|乐观参照，不能保证无语义模型相同|


预算不准入不等于证明完整实验实际一定超时。预检仅2个训练目标、一个临时早期模型，
其KMeans耗时随概率分布和聚类收敛变化，不能证明第1轮正式验证也需要同样时间。
本轮没有计数KMeans实际迭代次数，没有证据把耗时归因于固定某次数迭代或无语义模型的固有缺陷。
决策依据是用户要求的**保守估算准入**无法满足预算，不能据乐观旧模型耗时直接启动。
没有改采样、KMeans、K、验证规模、epoch或学习率来绕过预算。

原损失finite、梯度finite、真实K20完整路径、重复与逆序、原始未来反事实、checkpoint严格恢复后下一步一致均通过。

记录：preflight_results.json、preflight_steps.csv、preflight_gradients.csv、preflight_K20.csv。
