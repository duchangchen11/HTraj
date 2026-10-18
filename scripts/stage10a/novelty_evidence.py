"""Version and primary-paper evidence; no model execution or test metrics."""
from common import *


def main():
    cache = ROOT/'experiments/stage10a/cache'
    repo = cache/'Snapshot'
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    files = []
    for remote,local in [('utils/social_features.py','snapshot_social_features.py'),('config.yaml','snapshot_config.yaml'),('LICENSE','snapshot_LICENSE')]:
        blob = subprocess.check_output(['git','show',head+':'+remote],cwd=repo)
        assert blob == (cache/local).read_bytes()
        files.append(dict(repository='https://github.com/TUMFTM/Snapshot',commit=head,file=remote,sha256=sha(cache/local),
            url='https://github.com/TUMFTM/Snapshot/blob/'+head+'/'+remote))
    write_json(OUT/'source_versions.json',dict(base_commit='d8e753f718647db49202da3ac6d43e4b37cfc59b',
        execution_commit=commit(),Snapshot=dict(commit=head,license='Apache-2.0',no_checkout=True,blobs=files),
        local_reused_files=[dict(path=p,sha256=sha(ROOT/p)) for p in ['src/data/preprocessing.py','src/socialvae_causal/data.py']],
        models_imported_from_vendors=[],older_frozen_sources_unchanged='see history_before/after.json'))
    rows = [
        dict(name='Snapshot: Towards Application-centered Models for Pedestrian Trajectory Prediction in Urban Traffic Environments',year=2025,venue='WACV Workshops / HAVI',
            architecture='agent-centric social Transformer + vector-map cross-attention + CNN deterministic decoder',
            relation='t=-r dot v/||v||^2, d=||r+t*v||; exp(-.5*((t/delta_t)^2+(d/delta_d)^2+(bearing_rate/delta_a)^2)); top7',
            location='Section3.3 / Figure3 / Table3; utils/social_features.py calculate_cognitive_metrics, calculate_collision_risk, create_mover_matrix',
            overlap='DIRECT mechanism and question overlap: risk vs L2 neighbor selection already evaluated',
            difference='AV2 heterogeneous agents/map/1s->6s/single cutoff selection; Stage10 causal per-step finite horizon and GRU, not a novelty proof',
            paper='https://openaccess.thecvf.com/content/WACV2025W/HAVI/papers/Uhlemann_Snapshot_Towards_Application-centered_Models_for_Pedestrian_Trajectory_Prediction_in_Urban_WACVW_2025_paper.pdf',
            code='https://github.com/TUMFTM/Snapshot/tree/'+head,status='full methods and official source read, not executed'),
        dict(name='Higher-order Relational Reasoning for Pedestrian Trajectory Prediction',year=2024,venue='CVPR',
            architecture='collision-aware pair graph + graph powers + order fusion, plugged into forecasting backbone',
            relation='Aij=1/[dij*(||Cij-Qi||+||Cij-Qj||)] when motion rays intersect; Hm=Ahat^m X Wm',
            location='Sections4.1-4.2 / Eq2-5; Stage1 source register PDF SHA rechecked',
            overlap='collision-aware motion graph already published; relative-motion prior not new',
            difference='ray intersections not simultaneous finite-time CPA; Stage10 no high-order graph powers',
            paper='https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_Higher-order_Relational_Reasoning_for_Pedestrian_Trajectory_Prediction_CVPR_2024_paper.pdf',
            code='not re-verified in Stage10A; paper-level evidence only',status='full methods read; code URL NOT VERIFIED'),
        dict(name='SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction',year=2021,venue='CVPR',
            architecture='learned sparse directed spatial/temporal graphs + graph convolutions + bi-Gaussian output',
            relation='M=1[sigmoid(F)>=xi]; A=(M+I)*R; ZeroSoftmax(xi)=(exp(xi)-1)^2/(sum_j(exp(xj)-1)^2+eps)',
            location='Section3.1 / Eq1-5',overlap='adaptive sparse neighbors / directed motion relations already published',
            difference='learned asymmetric convolution mask, not analytic CPA topK; historic index/singleton bugs not reused',
            paper='https://www.chengjianglong.com/publications/SGCN_CVPR.pdf',
            code='https://github.com/shuaishiliu/SGCN/tree/5153702d8ae32a2b0586b093d4e094aee08b0f9b',status='methods read; prior source audit reused'),
        dict(name='Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction',year=2020,venue='CVPR',
            architecture='ST-GCN + TXP-CNN, bivariate Gaussian NLL',
            relation='Eq2 inverse L2 kernel on graph vertices; official maintenance loader seq_to_graph consumes relative displacements',
            location='Section4 / Section6.1 / Eq2,8; official frozen utils.py',
            overlap='dynamic graph kernels and relative displacement similarity already used',
            difference='Stage10 explicit risk topK, independent GRU/decoder, no node-index convolution or BN/PTN',
            paper='https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf',
            code='https://github.com/abduallahmohamed/Social-STGCNN/tree/333d3a57b4d2705e129b21aefefa09c79b2b9ae1',status='methods read; official old source audited'),
        dict(name='Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data',year=2020,venue='ECCV',
            architecture='dynamic typed graph, history/edge LSTMs, attention, CVAE and dynamics integration',
            relation='edge(i->j)=1[||pi-pj||<=perception_radius(type_j)]; sum neighbor states per edge type then LSTM',
            location='Section4.1 Modeling Agent History / Encoding Agent Interactions',
            overlap='distance dynamic graph, variable neighbor sets, relative states already standard',
            difference='Stage10 finite-time CPA selects topK; no typed recurrent edge/CVAE/dynamics model',
            paper='https://arxiv.org/abs/2001.03093',code='https://github.com/StanfordASL/Trajectron-plus-plus',status='full methods read; no new code execution/version comparison'),
        dict(name='SocialCircle: Learning the Angle-based Social Interaction Representation for Pedestrian Trajectory Prediction',year=2024,venue='CVPR',
            architecture='angular-partition meta-features embedded as virtual temporal sequence into backbone',
            relation='theta=atan2(pj_last-pi_last); per-sector means of neighbor net displacement magnitude, current distance and direction',
            location='Section3 / Eq1-9 and 12-13',overlap='position/motion/direction and neighbor aggregation already published',
            difference='sector aggregate is not CPA edge ranking; Stage9 pairing failure retained, not repackaged',
            paper='https://openaccess.thecvf.com/content/CVPR2024/papers/Wong_SocialCircle_Learning_the_Angle-based_Social_Interaction_Representation_for_Pedestrian_Trajectory_CVPR_2024_paper.pdf',
            code='https://github.com/cocoon2wong/SocialCircle',status='full methods read; code URL linked from official paper not cloned'),
    ]
    write_csv(OUT/'literature_comparison.csv',rows)
    write_json(OUT/'novelty_decision.json',dict(
        engineering_new_implementation=True,novel_core_risk_features=False,
        exact_end_to_end_equivalence_claimed=False,question_and_selector_overlap_with_Snapshot=True,
        preexisting_HTraj_CPA_ranking='src/models/temporal_graph.py:select_candidates closest+0.05*distance',
        research_GO=False,reason='No demonstrated predictive limitation or mathematically distinct contribution; conventional finite-horizon CPA ranking alone insufficient',
        implementation_success_not_prediction_success=True,formal_training_or_validation_error=False))


if __name__=='__main__':
    main()
