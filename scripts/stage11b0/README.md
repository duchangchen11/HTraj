# Stage11B-0：训练工程准入

只执行seed17、3个smoke更新、从同一随机初态重置的一次无增强完整Train遍历、两个配对恢复探针更新。没有正式多轮训练、没有验证或测试精度、没有新算法。

采用隔离的`experiments/stage11ap/cache/venv/bin/python`，不安装/降级旧包。所有本轮权重、过滤后Train数据、官方源副本与论文补充包在忽略的`experiments/stage11b0/cache/`，只有代码/配置/报告/轻量结果入Git。

实际执行顺序：

1. `python -m scripts.stage11b0.protect --freeze`：继承旧保护表，增加Stage11A+全部源/资源/权重。
2. `data_audit`：核对源manifest、Train/Val GT SHA、录像支持；Test仅opaque文件SHA、不反序列化。之后物化只含7496 Train目标的本阶段cache pkl。
3. `runtime`只安全加载作者分割权重，严格不加载作者轨迹预测权重；原YNet卷积核心随机初始化。`training_adapter`实现原两个BCE/teacher forcing与独立数据遍历，冻结分割器eval/BN。
4. 前置12项CPU测试；首次路径记录错误及7个setup error原始日志以gzip无损保留，没有跳过失败。修复只涉及保存路径元数据。
5. `freeze_config`、Git提交配置源码，再`run_admission`。固定scene批次1/target批次4，先测真实1/4/4，再恢复随机初态完成7496目标一次；所有真实step有逐条CSV。
6. `post_audit`只做CPU计数/参数差异/预算外推，不增加GPU训练；8倍增强不是本轮已运行训练。
7. 补充材料4.1/4.2的新增证据在semantic_supplement_addendum；类别编号与当前分割checkpoint成员仍未核实。
8. 全仓库pytest，保护历史SHA，交付SHA索引、diff检查、最终commit。

完整测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
experiments/stage11ap/cache/venv/bin/python -m pytest tests \
scripts/stage11a/tests scripts/stage11ap/tests scripts/stage11b0/tests -q
```

原官方训练损失源为Human-Path-Prediction/ynet/train.py，MIT，commit a345ada1557f53dc2acb4c07f71d340d366980da。复用的Stage11A+受限pickle、旧式分割checkpoint loader与数据增强源均记录旧SHA，没有改动。checkpoint为state_dict、Adam、RNG、配置，恢复使用weights_only=True/strict=True；原方法没有scheduler，显式记录None。

Scene代表录像轨迹容器，目标数不代表同时可见人数。两项原出界点完整保留；补边后热图有效。GT合法参与训练监督与teacher forcing，独立推理API没有future参数。全部工程计时是实际训练，不取旧无梯度推理作训练成本。

工程GO与正式协议冻结、预算批准、SCI创新性分开；完成后停止。
