"""Evidence-indexed manual literature review; not a claim of exhaustive novelty."""
from .common import *
import re
import shutil

# Facts transcribed from the downloaded primary methods, not search snippets.
# A loaded full text is not automatically treated as a verified method: anchors below identify what was read.
PAPERS=[
 ('YNet','From Goals, Waypoints & Paths to Long Term Human Trajectory Forecasting','Karttikeya Mangalam; Yang An; Harshayu Girase; Jitendra Malik',2021,'ICCV 2021; methods source arXiv v1 (2020)',
  'https://arxiv.org/abs/2012.01526','https://karttikeya.github.io/publication/ynet/',
  '行人多模态轨迹','历史坐标+场景图像','Goal/waypoint热图+完整路径','是','Sigmoid像素权重后离散采样','TTST:10000 -> K-1 KMeans + 1 Softargmax',
  '明确按有限K采样','训练网络；TTST无额外训练','GT监督热图','否','SDD; ETH/UCY; inD','ADE/FDE best-of-K',
  '本项目完整生成链路来源','现研究冻结链路的局部失覆盖，不是原论文新模型','HIGH','Test-Time Sampling Trick',
  '官方冻结代码a345ada；相对阈值和KMeans源码已读；全文为较早arXiv版本，不冒充最终版逐字相同'),
 ('PECNet','It Is Not the Journey but the Destination: Endpoint Conditioned Trajectory Prediction','Karttikeya Mangalam; Harshayu Girase; Shreyas Agarwal; Kuan-Hui Lee; Ehsan Adeli; Jitendra Malik; Adrien Gaidon',2020,'ECCV 2020',
  'https://arxiv.org/abs/2004.02025','https://karttikeya.github.io/publication/htf/',
  '行人轨迹','历史+社会邻居','端点CVAE+条件轨迹','是','隐式潜分布','核心截断trick不是聚类','调整少样本精度/多样性','需要训练','GT端点/轨迹监督','否','ETH/UCY; SDD','ADE/FDE best-of-K',
  '有限候选与概率多样性取舍','潜空间方差/截断，不是冻结proposal压缩','HIGH','Truncation trick','代码仅通过作者项目定位，未执行'),
 ('DLow','DLow: Diversifying Latent Flows for Diverse Human Motion Prediction','Ye Yuan; Kris Kitani',2020,'ECCV 2020',
  'https://arxiv.org/abs/2003.08386','https://www.ye-yuan.com/dlow',
  '3D人体动作多模态预测','过去关节运动','相关潜码解码K种未来','非显式2D Goal','潜高斯+KL','不是KMeans后处理','有限K多样性/似然取舍','需要训练采样映射','原CVAE和重建项使用GT','否','Human3.6M; HumanEva-I','APD; ADE/FDE; MMADE/MMFDE',
  '有限候选多样性不是新问题','动作潜空间，不是2D端点采样云','HIGH','Latent Mapping Functions','K头仿射共享噪声；RBF成对能量+KL；与DPP论文分开'),
 ('AgentFormer','AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting','Ye Yuan; Xinshuo Weng; Yanglan Ou; Kris Kitani',2021,'ICCV 2021',
  'https://arxiv.org/abs/2103.14023','https://github.com/Khrylx/AgentFormer',
  '多行人社会时空预测','多行人历史+可选地图','CVAE+DLow完整轨迹','无独立Goal热图','学习先验与潜高斯','无TTST KMeans核心链路','DLow联合覆盖与多样性','CVAE及DLow两阶段','后验/重建训练用GT','否','ETH/UCY; nuScenes','minADE/minFDE K20',
  '历史条件有限候选与DLow已有','修改完整模型不是固定proposal对照','HIGH','Trajectory Sampler','已读CVAE/式7及采样器式9；不重算已有测试指标'),
 ('TUTR','Trajectory Unified Transformer for Pedestrian Trajectory Prediction','Liushuai Shi; Le Wang; Sanping Zhou; Gang Hua',2023,'ICCV 2023',
  'https://openaccess.thecvf.com/content/ICCV2023/html/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.html','未在本次证据中核验官方代码链接',
  '行人轨迹','历史+邻居+训练集运动模式','并行轨迹与概率','轨迹模式而非Goal像素','显式输出模式概率','训练前聚类模式；推理不聚类','明确研究聚类开销和概率丢失','需要训练','GT回归与分类监督','否','ETH/UCY; SDD','ADE/FDE; brier-ADE/FDE',
  '聚类后处理概率丢失已有明确问题定义','模型级并行输出，不是固定采样云代表点','HIGH','Global Prediction','全文§3；训练模式聚类不可误写为完全无聚类'),
 ('DPP','Diverse Trajectory Forecasting with Determinantal Point Processes','Ye Yuan; Kris M. Kitani',2020,'ICLR 2020',
  'https://arxiv.org/abs/1907.04967','https://www.ri.cmu.edu/app/uploads/2020/04/Yuan_DPP_forecast.pdf',
  '2D轨迹与3D动作','历史条件特征','DSF潜码与多样轨迹集','无独立目标热图','质量-相似度DPP核','非KMeans后处理；可做集合选择','显式处理多数/少数模式与有限集','需要训练DSF','训练CVAE/任务监督；DPP损失基于生成集','否','合成2D轨迹; 自采动作; Human3.6M','多样性; ADE/FDE; 多未来评价',
  '质量/概率和多样性权衡直接重合','学习潜码集合，不是已冻结19簇内替换代表','HIGH','Quality.','方法§4，核Diag(r)S Diag(r)；质量不是直接用轨迹似然以免只保大模式'),
 ('DenseTNT','DenseTNT: End-to-End Trajectory Prediction From Dense Goal Sets','Junru Gu; Chen Sun; Hang Zhao',2021,'ICCV 2021',
  'https://arxiv.org/abs/2108.09640','https://tsinghua-mars-lab.github.io/DenseTNT/',
  '自动驾驶轨迹','目标历史+道路向量','稠密Goal分布到有限Goal/路径集','是','稠密Goal概率h(c)','优化/集合预测替代NMS','直接最小化概率加权集合期望误差','完整模型训练；离线优化伪标签','GT训练热图；伪目标用预测分布','否','Argoverse; Waymo','minADE/minFDE; MR',
  '概率加权有限K代表性问题直接先例','地图车辆与可训练集合头，不是SDD冻结cloud','HIGH','Optimization for Offline Goal Set Prediction','方法§3；f(Y)=sum h(c) min_y ||c-y||；不能把该目标包装新颖'),
 ('NPSN','Non-Probability Sampling Network for Stochastic Human Trajectory Prediction','Inhwan Bae; Jin-Hwi Park; Hae-Gon Jeon',2022,'CVPR 2022',
  'https://arxiv.org/abs/2203.13471','https://github.com/inhwanbae/NPSN',
  '行人随机轨迹','目标/邻居过去路径','N个目的性潜采样/轨迹','不限定Goal像素','QMC或学习潜样本','无必要KMeans后处理','明确有限样本覆盖偏置','QMC无需训练；NPSN需要训练','min轨迹距离监督+discrepancy','否','ETH/UCY; SDD; GCS','ADE/FDE; TCC',
  '历史运动引导候选/低差异采样已有','新采样而非保留既有10000点','HIGH','NPSN Architecture','§4 式8/9、QMC与Box-Muller；不可称引入历史状态即新'),
 ('HYPER','HYPER: Learned Hybrid Trajectory Prediction via Factored Inference and Adaptive Sampling','Xin Huang; Guy Rosman; Igor Gilitschenski; Ashkan Jasour; Stephen G. McGill; John J. Leonard; Brian C. Williams',2022,'ICRA 2022; arXiv v1 2021',
  'https://arxiv.org/abs/2110.02344','https://tisl.cs.toronto.edu/publications/HYPER%3A%20Learned%20Hybrid%20Trajectory%20Prediction%20via%20Factored%20Inference%20and%20Adaptive%20Sampling',
  '车辆多模态轨迹','过去轨迹+道路图','混合意图轨迹样本及似然','用终点距离选择','显式样本似然','最高似然初始化FPS选终点分离样本','明确有限样本覆盖/概率保留','训练混合模型；FPS不训练','MLE/提议分布训练GT','否','Argoverse v1.1','minADE/minFDE; coverage',
  '直接使用FPS进行预测样本压缩','车辆混合意图，不是YNet热图聚类','HIGH','Trajectory Sample Selection','§IV-D/V；FPS已经用于轨迹预测，不能声称只属于数学背景'),
 ('LRVQ2024','Trajectory Forecasting through Low-Rank Adaptation of Discrete Latent Codes','Riccardo Benaglia; Angelo Porrello; Pietro Buzzega; Simone Calderara; Rita Cucchiara',2024,'ICPR 2024 (author official README); downloaded arXiv v2',
  'https://arxiv.org/abs/2405.20743','https://github.com/aimagelab/LRVQ',
  '行人及运动轨迹','历史+社会上下文','VQ-VAE/扩散轨迹到K中心','全路径而非Goal点','隐式扩散先验','N=200路径 -> K=20 k-means质心','明确精度/探索压缩取舍','两阶段训练；质心选择不训练','GT训练重建/扩散','否','SDD; NBA; NFL','best-of-20 ADE/FDE',
  '2024仍直接研究过采样到KMeans压缩','完整轨迹聚类，非10000Goal19+Soft','HIGH','Enforcing Effective Multi-modal Forecasting','§4.4：实际方法全文，不只摘要；质心均值降量化噪声的论点需实验对照'),
 ('GoalFlow2025','GoalFlow: Goal-Driven Flow Matching for Multimodal Trajectories Generation in End-to-End Autonomous Driving','Zebin Xing; Xingyu Zhang; Yang Hu; Bo Jiang; Tong He; Qian Zhang; Xiaoxiao Long; Wei Yin',2025,'CVPR 2025; downloaded arXiv v6 2025-10-01',
  'https://arxiv.org/abs/2503.05689','https://github.com/YvanYin/GoalFlow',
  '自车端到端规划，不是行人预测','传感器BEV+自车状态','Goal引导flow候选与评分','是','候选分数，不等同校准Goal热图','非YNet TTST19簇','候选生成/选择，但任务不同','需要训练','GT轨迹/Goal与评分监督','否','NAVSIM','PDMS，不是行人ADE基准',
  '场景条件Goal与候选选择相近','任务/输入/指标不同，不冒充直接行人同题','MEDIUM','Goal Point Scorer','已读Goal构建和规划模块；不作为当前预测指标比较'),
 ('GMM2026','Rethinking Training & Inference for Forecasting: Linking Winner-Take-All back to GMMs','Qiyuan Wu; Katie Z Luo; Bharath Hariharan; Wei-Lun Chao; Mark Campbell',2026,'作者arXiv注明Accepted by ECCV 2026；最终proceedings未逐字核验',
  'https://arxiv.org/abs/2606.26424','本次未核验官方代码发布链接',
  '自动驾驶多模态预测','既有模式轨迹与后验概率','合并后的有限模式','使用终点作分配距离','显式模式概率','测试时概率加权KMeans合并','直接研究有限模式压缩与概率失配','合并无需训练；一步EM需训练集M-step','原训练/EM使用训练GT；测试合并不用GT','否','nuScenes; Waymo Open Motion','minADE/minFDE; MR; brier-minFDE',
  '概率保持/加权聚类2026直接冲突','WTA模式而非重复离散Goal样本；并非完全等价','HIGH','Test-Time Merging','§5.1：按终点分配，全轨迹按模式后验加权均值；不能声称概率加权为原创')
]

def main():
    cpu_guard();fetch=json.loads((OUT/'literature_fetch.json').read_text());fetched={r['paper']:r for r in fetch['records'] if 'PDF_SHA' in r}
    fields=['key','title','authors','year','venue','official_link','code_or_project','task','inputs','outputs','uses_goal',
      'uses_probability','clustering_postprocessing','finite_K_coverage','extra_training','GT_training','GT_inference','datasets','metrics',
      'overlap','difference','novelty_risk','method_anchor','notes']
    rows=[]
    for values in PAPERS:
        r=dict(zip(fields,values));txt=CACHE/'literature'/(r['key']+'.txt');text=txt.read_text();anchor=r['method_anchor']
        matches=[i+1 for i,l in enumerate(text.splitlines()) if anchor.casefold() in l.casefold()]
        assert matches,(r['key'],anchor)
        r.update(evidence_level='FULLTEXT_METHOD_REVIEWED',method_text_lines=';'.join(map(str,matches[:5])),
            PDF_SHA=fetched[r['key']]['PDF_SHA'],text_SHA=sha(txt),fulltext_version=next((l.strip() for l in text.splitlines() if 'arXiv:' in l),'CVF official PDF'),
            inference_GT_status='NO_INFERENCE_GT; no implementations executed this stage',review_limit='scoped review, not exhaustive originality proof')
        rows.append(r)
    csvnew(OUT/'literature_review.csv',rows)
    matrix=[
      ('A','直接替换KMeans19','YNet;LRVQ2024;HYPER','YES','YES','k-means/medoid/FPS是常规方法','EXISTING_CACHE','不改网络；改变Goal需重解码','YES','NO','86有失覆盖；不证明替代有效','HIGH','固定分区内已有样本代表点对照'),
      ('B','增加候选多样性','DLow;DPP;NPSN;AgentFormer','YES','YES','DPP/RBF距离/低差异采样已有','HISTORY_AND_PREDICTIONS','通常改变采样或训练','YES','方法相关','几何离散度不等于性能收益','HIGH','同K同时检查覆盖与完整路径，不新增K'),
      ('C','按概率质量调整代表性','DenseTNT;GMM2026;TUTR','YES','YES','概率加权集合误差/weighted KMeans已有','PREDICTED_PROBABILITY','可只改后处理','YES','可不训练','低质量局部被遗漏；未证明权重重复使用合理','HIGH','比较经验频数权重与显式概率，防二次加权'),
      ('D','高低概率区域代表性','DPP;NPSN;HYPER;DenseTNT','RELATED','RELATED','覆盖/质量-多样性取舍已有','PREDICTED_PROBABILITY','可只改后处理','YES','可不训练','失败组低局部质量；保护所有尾部可能保噪声','HIGH','与普通覆盖选择同预算比较；不是新机制准入'),
      ('E','历史运动条件压缩','NPSN;TUTR;LRVQ2024','YES_RELATED','YES_RELATED','历史条件采样/实例码本已有','HISTORY_ONLY','需额外条件函数或规则','YES','取决于方法','仅有事后分组关联，未证独立收益','HIGH','先按录像复核条件关联；不直接加新模块'),
      ('F','语义可靠性候选生成','YNet;GoalFlow2025','RELATED_NOT_IDENTICAL','UNVERIFIED','场景Goal已有；可靠性因果机制未核实','SCENE_HISTORY','会改变输入/Goal或选择','YES','可能','本阶段没有语义致错直接证据','UNKNOWN','须先验证语义反事实；当前不准入')]
    csvnew(OUT/'literature_overlap_matrix.csv',[dict(zip(['direction','proposal','closest_work','same_problem','similar_objective','mathematical_prior',
        'deployment_input','change_YNet','fixed_K20_fair','extra_training','current_evidence','innovation_risk','minimum_falsifiable_control'],r)) for r in matrix])
    math=[
      dict(method='k-center / FPS',objective='min_C max_j min_c ||P_j-c||',prior='HYPER §IV-D uses endpoint FPS in trajectory selection',
           official_link='https://arxiv.org/abs/2110.02344',direct_trajectory_use='YES',limit='worst-case geometric coverage != GT hit/ADE; FPS2-approx requires metric and suitable domain'),
      dict(method='k-medoids',objective='min_{c in observed samples} sum_j min_c distance(P_j,c)',prior='Kaufman & Rousseeuw, Finding Groups in Data (1990), partitioning around medoids',
           official_link='https://doi.org/10.1002/9780470316801',direct_trajectory_use='NOT_CLAIMED_FROM_THIS_REFERENCE',limit='unsquared and squared objectives differ; no originality claim'),
      dict(method='probability-weighted representative set',objective='sum_x p(x) min_c ||x-c||',prior='DenseTNT Eq6/Algorithm1',
           official_link='https://arxiv.org/abs/2108.09640',direct_trajectory_use='YES',limit='prior predicted p may be miscalibrated; does not guarantee local GT coverage'),
      dict(method='DPP quality/diversity',objective='L=Diag(r) S Diag(r); expected-cardinality diversity objective',prior='Yuan & Kitani ICLR2020 §4',
           official_link='https://arxiv.org/abs/1907.04967',direct_trajectory_use='YES',limit='quality of minor modes already discussed; not a new name for FPS'),
      dict(method='quantization / rate-distortion',objective='KMeans empirical E[min ||P-c||^2] vs radius coverage E[1(min ||P-c||<=r)]',prior='Lloyd, Least Squares Quantization in PCM, IEEE TIT 1982; connection is mathematical background',
           official_link='https://doi.org/10.1109/TIT.1982.1056489',direct_trajectory_use='LRVQ2024 uses quantization and KMeans; not this classic alone',limit='not an information-theoretic rate-distortion theorem for this YNet experiment')]
    csvnew(OUT/'mathematical_prior_art.csv',math)
    for old,name in [('/tmp/htraj_stage11b2d_openalex.txt','discovery_runtime_failure.txt'),('/tmp/htraj_stage11b2d_openalex_retry.txt','discovery_OpenAlex_failure.txt')]:
        shutil.copyfile(old,OUT/name)
    jnew(OUT/'literature_search_audit.json',dict(date='2026-10-22',workflow='multi-source-search',
      academic_MCP='NOT_MOUNTED',fallback=['primary CVF/arXiv/author web search','OpenAlex discovery script'],
      failures=['system Python3.8 incompatible with skill type annotations; retried using existing Python3.10',
                'OpenAlex HTTP429; no repeated retry; primary papers accessible'],
      fulltext_reviewed=len(rows),recent=['LRVQ2024','GoalFlow2025','GMM2026'],
      queries=json.loads((CONF/'analysis_protocol.json').read_text())['literature_scope']['queries'],
      additional_queries=['trajectory prediction farthest point sampling','trajectory prediction weighted clustering 2025',
        'Rethinking Training Inference Forecasting 2026','Non-Probability Sampling Network','trajectory prediction k-medoids'],
      duplicate_handling='deduplicated by arXiv ID/title; preprint and conference counted once; DLow != DPP',
      excluded_ambiguity='Search found two different SIPTraj descriptions. Downloaded arXiv2608.00779 is map-free autonomous driving and not evidence for the thesis weighted-FPS claim; no conflation.',
      literature_limit='Not exhaustive; no proof of originality; no patent freedom-to-operate opinion',
      new_models_run=0,model_code_unchanged=True))
    print('LITERATURE',len(rows),'primary fulltext methods; overlap matrix six directions',flush=True)

if __name__=='__main__':main()
