# 官方权重核验（评价前冻结）

沿用Stage7.2已严格加载的唯一作者压缩包，SHA256：`f192aa9596c25ec115dbf646f96a03f5b17e2c969016cd1b9c7e753db30941f0`。不加载smoke权重，不重新选择epoch。

|折|CVAE索引|DLow索引|traj_scale|合计参数（含未注册CVAE）|
|---|---:|---:|---:|---:|
|ETH|30|5|2|7,089,794|
|HOTEL|50|50|1|6,563,970|
|UNIV|70|15|1|6,563,970|
|ZARA1|100|40|1|6,563,970|
|ZARA2|70|25|1|6,563,970|

全部10个原始配置字节复制到`configs/stage7p3/official/`，内容SHA必须与官方源文件相同。完整checkpoint路径、文件SHA、原配置SHA及二者索引关系记入`evaluation_protocol.json`的pairs与checkpoints_and_configs。

预检将再次严格恢复DLow，并逐张量确认其内部CVAE与指定CVAE权重完全一致，检查所有参数有限及官方测试序列入口。此预检不预测测试轨迹、不计算测试误差。真实预检结果见`results/tables/stage7p3/preflight.json`。

来源：`results/tables/stage7p2/official_checkpoint_acquisition.json`与`official_checkpoint_strict_loading.json`；[作者模型说明](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5#pretrained-models)。

Git空白检查例外：官方YAML带有末尾空行。为保留其原始字节和SHA，仅`configs/stage7p3/official/.gitattributes`对这些原样副本关闭空白诊断；其余新增文件仍执行正常diff检查。首次冻结提交在检查时被阻止，未启动推理；设置局部属性后重新检查和提交。
