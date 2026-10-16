"""Append measured acceptance details after final execution; preserve old stages."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import json
from src.socialvae_causal.io import read_csv
from src.socialvae_baseline.artifacts import json_new
from src.data.preprocessing import sha256
root=Path('results/tables/stage5p4');report=Path('reports/stage5p4')
v=json.load(open(root/'delivery_verification.json'))
r=read_csv(root/'combined_capacity_audit_final.csv')
text='\n## 最终真实验收\n\n'+f"完整测试{v['unit_tests']}项通过；历史保护{v['protected_files_verified']}文件SHA未变。第一阶段结果与checkpoint在第二阶段前另建保护清单且核验未变。四组正式run均100完整epoch；未读ETH指标。\n\n"
text+='|模型|样本范围|名义参数|loss连接模块参数|结构有效上界|实测非零梯度标量|未观测非零梯度标量|\n|---|---|---|---|---|---|---|\n'
for row in r:
 text+='|'+'|'.join(row[k] or '未记录' for k in ['model','recording','nominal','module_loss_connected','structural_upper_bound','observed_nonzero','observed_zero'])+'|\n'
text+='''
64目标probe为4批只前后向、0优化更新；录像行是100轮梯度并集。loss连接模块参数含恒零输入列，结构上界另外排除这些标量；实际非零梯度受数据和激活影响。长期未观测非零不等于永久理论死参数。Full历史训练没有逐标量梯度并集，不能从checkpoint精确倒推该数；旧Adam二阶矩仅是代理，未填成精确值。参数数量接近不代表函数类/秩/归纳偏置相同。

训练后：EgoCapacity与Placebo各自最佳checkpoint均通过真实历史目标上的位置、速度、数量、顺序、进入时间干预；K1、公共噪声K20、监督后验与原始loss全部严格不变，原始邻居输入Jacobian无依赖；有限差分也为0。原Full的no_grad特征构造不等价于无邻居依赖，本阶段不借此抹去其社会输入功能。诊断未更新模型权重。

所有原始单测XML/TXT、每轮参数梯度、训练后隔离/数值Jacobian、checkpoint恢复和来源SHA均保留。真实数据manifest、目标历史/GT SHA和公共采样seed与冻结Full/NoSocial一致。Placebo社会路径实际执行、各社会模块获得非零梯度，但恒定初始偏移与合成分布仍限制容量解释。
'''
with (report/'02_CAPACITY_AND_ISOLATION_TESTS.md').open('a') as f:f.write(text)
for variant in ['placebo_social']:
 for rec in ['biwi_hotel','crowds_zara02']:
  p=root/variant/rec/'execution.jsonl'
  with p.with_suffix('.txt').open('x') as f:f.write(p.read_text())
files=[str(p) for d in ['src/socialvae_capacity','scripts/stage5p4'] for p in Path(d).glob('*.py')]
files+=['tests/test_socialvae_capacity.py','tests/test_socialvae_capacity_decision.py','tests/test_socialvae_placebo.py','configs/stage5p4/ego_protocol.json','configs/stage5p4/placebo_protocol.json']
json_new(root/'final_analysis_code_sha.json',{p:sha256(p) for p in sorted(files)})
print('Measured capacity, isolation, acceptance and source-version details appended')
