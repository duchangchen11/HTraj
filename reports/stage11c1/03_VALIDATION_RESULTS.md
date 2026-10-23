# 验证状态

7次固定998目标验证均**未执行**；没有NoSemantic ADE/FDE或GoalCover，也没有best或last正式checkpoint。

原语义与新无语义没有形成数值对照。validation_metrics.csv仅表头，不能把空值当零。

冻结语义Epoch30仅作为只读历史参照：ADE1=32.093099789516px，FDE1=60.539020848913px，
minADE20=16.403135768685px，minFDE20=25.634029077877px，GoalCover25=834/998。
这些值来源Stage11B-1C原始training_execution.json及配对结果，**不是本轮无语义成绩**。

图1为冻结协议示意；其余11项需要真实训练/验证/配对预测，未生成。预算与预检耗时图仅显示实测与标注估算，见figure_manifest.json。
