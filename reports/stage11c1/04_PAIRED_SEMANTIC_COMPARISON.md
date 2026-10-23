# 配对比较状态

**NOT_EVALUATED**。没有无语义预测，不能计算固定Epoch30或best比较。

新输入14通道=6全零语义+8原始历史距离热图。语义float32全画布含padding为零；
原Goal与Path decoder、softargmax、损失及GT waypoint训练teacher forcing保持。
验证接口只接收冻结history-only缓存，不接收GT；定义为先保存998条K20结果并校验完整性，再读取GT评分。
K20仍为原CPU multinomial每sample_id独立SHA种子、10000有放回proposal、原19中心KMeans与1个softargmax。
模型选择仍为完整验证中窗口加权minADE20最低，完全相同则较早epoch；配对FDE同一checkpoint。
科研主比较固定Epoch30对Epoch30，五录像等权macro；best比较单列次要分析，不能事后挑有利方案。
科研门槛分母统一NoSemantic：macro ADE改善≥2%，macro FDE语义退化≤1%，至少3/5录像ADE支持语义，
OTHER4合计语义ADE退化≤1%。最后一项把用户“明显反向”事前具体化为1%，未依据验证误差选择。
Delta_ADE/FDE=NoSemantic−Semantic；Delta_Cover=Semantic−NoSemantic，正值均支持语义。


原语义失败164、成功834仅冻结为参照；无语义恢复、丢失、两者共同失败均未知。所需比较CSV为空schema或显式NOT_EVALUATED，不虚构样本行。

冻结语义Epoch30仅作为只读历史参照：ADE1=32.093099789516px，FDE1=60.539020848913px，
minADE20=16.403135768685px，minFDE20=25.634029077877px，GoalCover25=834/998。
这些值来源Stage11B-1C原始training_execution.json及配对结果，**不是本轮无语义成绩**。
