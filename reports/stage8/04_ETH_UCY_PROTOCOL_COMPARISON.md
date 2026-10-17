# ETH/UCY协议可比性

标准主目标固定五场景留一、8/12、0.4秒、米、K20，minADE与minFDE各自选最优候选。论文名称相同或时间长度相同不足以构成严格公平比较。

|项目|AgentFormer既有官方协议|Resonance原发布|MoFlow原发布|
|---|---|---|---|
|source|Frozen74 official17-column files,8recordings|Luna true_pos_.csv -> custom config/annotations; no74-file identity equivalence|SocialGAN/EigenTrajectory original pickle vs LED npy; two versions|
|window|current frame grid, contiguous8+12, includesN1|per-target track start/stop, step/interval; different indices; native partial-neighbor padding|original generator sorts unique frames20 and min_ped1 strict>1, so single-person windows excluded|
|context|Complete20 valid_id, all legal selected nodes|Visible at current obs; nearest cap; missing prefixes/suffixes padded; not fullfuture-neighbor qualification|original all_data[:,None] makesA1; LED format needs separate qualification audit|
|split|Separate official train/val/test sequence lists|Luna get_splits val=test; UNIV13 groupsuniv+univ3|train/test-only loading by main scripts; trainer val_loader defaults test_loader|
|ETH sampling|Frozen official ETH file grid/scale2; outputmetres|Author eth raw6frames, otherclips10; eth10 alternative; native declared0.4 not cohort equality|Chooseoriginal/LED explicitly; normalization fromtrain; no alignment by name alone|
|UNIV|students001+students003 target-count weighted|univ13 combinesuniv/univ3; exact ID/frame map unverified|raw/original/LED files need explicit source mapping and count check|
|K20|DLow20 mean=True latent heads|K1 repeats x Kc20 learned styles with random self/re-bias noise|20 correlated flow queries, teacher multistep or separately trained one-step student|
|metrics|minADE20/minFDE20 independently, targetweighted then5foldmacro|ADE_2D min acrossK, FDE ownlaststep; cohort differs|separate minADE/minFDE over20; end4.8s; cohort differs|
|current strict comparability|Retained reproduced reference, oldtestnotusedselection|NO direct ranking against AgentFormer|NO direct ranking against AgentFormer/Re|


## ETH时标与坐标

Luna配置eth=[6,25]、其他录像[10,25]，作者解释ETH视频被加速，6帧间隔用于其物理时长对齐，并提供eth10替代脚本。本轮不擅自裁决哪个时间解释“正确”，也不据此重算测试。关键事实是原始索引间隔/窗口及目标不同，必须核验帧时标和GT身份。AgentFormer维护版ETH还需traj_scale2还原；Luna用自有order/scale1；MoFlow按训练统计min-max再还原/历史朝向旋转。单靠米制不能保证坐标与测试对象一致。

Luna univ13把univ与univ3一起测试，并排除二者训练；其alias到students001/003的逐ID/frame映射未完整核验。AgentFormer明确两条学生序列按目标数汇总。不能只因两个文件就假定完全一致。

## 合法适配方案（未启动训练）

后续若重新立项，可保持AgentFormer已冻结74文件与独立train/val/test清单，先导出稳定(sequence,frame,personID)与GT SHA。Re每个目标用同一C20上下文，输入邻居相对目标最后位置；保留所有合法目标和拥挤窗口，不因单人/高密度删样本。MoFlow若A1仍不引入社会关系，可在同一目标集合训练个体生成参考；如要改为多人，数据资格、固定人数/排列编码和完整训练均需重新审查，不能作为原结构同协议现成社会基线。

所有学习基线必须在该冻结协议从头训练，显式独立validation选模；模型loss/输入语义与作者原发布的差异逐项报告。保留Resonance原协议结果为来源对照，不能把原权重与新协议结果直接排名。CV-Last2/OLS4/OLS8用同目标，明确它们只有一条确定性轨迹，不与K20构成相同候选预算证明。

本轮仅构建了HOTEL-fold validation的小量Re输入适配，688窗口、5203目标窗口，不是训练池/五折loader已全部验证。对应数据可与作者预训练模型来源训练重叠，工程前向不能作泛化评价。旧AgentFormer test已经被观察，此后也不能称其为完全未接触独立确认集；本轮不读取旧测试成绩作选择、不重新计算测试误差。

因此当前无法发布严格可比的三方法成绩排名。存在可解释的适配方案，不等于该方案已经冻结、训练和双评测验证完成。
