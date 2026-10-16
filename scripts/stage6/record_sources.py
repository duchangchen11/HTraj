"""Primary-source metadata, explicit search degradation, and bounded citations."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json,concurrent.futures
from scripts.stage6.fetch_public import request
from src.socialvae_baseline.artifacts import json_new

root=Path('results/tables/stage6/literature');root.mkdir(parents=True)
urls=[('arxiv_metadata.xml','https://export.arxiv.org/api/query?id_list=2104.01528,2403.18452,2203.05046&max_results=3'),
 ('sgcn_crossref.json','https://api.crossref.org/works/10.1109/CVPR46437.2021.00888')]
def get(item):
 name,url=item;data,log=request(url)
 if data is not None:
  with (root/name).open('xb') as f:f.write(data)
 return log
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:logs=list(pool.map(get,urls))
sources=[
 dict(id='SGCN',title='SGCN: Sparse Graph Convolution Network for Pedestrian Trajectory Prediction',year=2021,venue='CVPR',url='https://openaccess.thecvf.com/content/CVPR2021/html/Shi_SGCN_Sparse_Graph_Convolution_Network_for_Pedestrian_Trajectory_Prediction_CVPR_2021_paper.html',arxiv='2104.01528',doi='10.1109/CVPR46437.2021.00888',claim='Sparse directed spatial and temporal graphs are existing published ideas.',status='CVF/arXiv/Git source cross-check'),
 dict(id='Social-STGCNN',title='Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction',year=2020,venue='CVPR',url='https://openaccess.thecvf.com/content_CVPR_2020/html/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.html',claim='Existing social graph baseline. Local tests concern maintained commit, not exact 2020 code.',status='CVF plus previous frozen local audits'),
 dict(id='SingularTrajectory',title='SingularTrajectory: Universal Trajectory Predictor Using Diffusion Model',year=2024,venue='CVPR',url='https://openaccess.thecvf.com/content/CVPR2024/html/Bae_SingularTrajectory_Universal_Trajectory_Predictor_Using_Diffusion_Model_CVPR_2024_paper.html',arxiv='2403.18452',claim='Motion representation unification and adaptive anchors already exist; domain adaptation is not target-free domain generalization.',status='CVF/arXiv and fixed Git source'),
 dict(id='MotionPriorDG',title='Towards Single Source Domain Generalisation in Trajectory Prediction: A Motion Prior based Approach',year=2023,venue='CoLLAs / PMLR232:227-243',url='https://proceedings.mlr.press/v232/huang23a.html',claim='CV-prior residuals and cross-domain motion prediction are existing ideas, not HTraj novelty.',status='Publisher primary abstract/metadata; not reproduced, not baseline candidate'),
 dict(id='T-GNN',title='Adaptive Trajectory Prediction via Transferable GNN',year=2022,venue='CVPR',url='https://arxiv.org/abs/2203.05046v2',claim='Domain alignment and transferable GNN are prior work; adaptation access assumptions must be separated from target-free generalization.',status='Author arXiv metadata; not reproduced, not additional baseline candidate')]
json_new(root/'sources.json',dict(sources=sources,metadata_requests=logs,workflow='nature-academic-search / multi-source-search',
 academic_search_MCP_available=False,OpenAlex_fallback='HTTP429, no blind retry; original log in experiments/stage6/academic_search.txt',
 GitHub_REST='HTTP403, source pinning used Git plus SHA-verified raw blobs',search_scope='Focused primary-source reassessment, not exhaustive systematic review or a novelty certificate',
 ETH_scores_used=False,unverified_performance_claims=False))
print('Primary metadata and search failure provenance saved')
