"""Seal measured budget metadata before training; no validation errors accessed."""
import csv,gzip,json
from common import *
with gzip.open(OUT/'train_targets.csv.gz','rt') as f:targets=list(csv.DictReader(f))
order=np.random.default_rng(9001).permutation(len(targets))
with (OUT/'timing_steps.csv').open() as f:timing=list(csv.DictReader(f))
for r in timing:
    s=int(r['step'])-1
    r['actual_max_context']=max(int(targets[int(i)]['context_people']) for i in order[s*128:(s+1)*128])
    r['context_instrumentation_note']='Original max_context after step1 used padded upper bound; actual_max_context is verified against frozen batch IDs'
csv_new(OUT/'timing_context_verified.csv',timing)
prep=(OUT/'data_manifest.json').stat().st_mtime-(OUT/'data_preparation.txt').stat().st_mtime+2
# Empty prep log mtime marks shell creation before startup, manifest mtime marks end.
gpu_ledger('data_preparation_conservative_wall_from_file_timestamps',prep)
b=json.loads((OUT/'budget_estimate.json').read_text());t=json.loads((OUT/'timing.json').read_text());c=json.loads((ROOT/'configs/stage9/training.json').read_text())
report='# 真实测速与冻结预算\n\n未计算验证ADE/FDE。A/B/C各50次真实optimizer更新，测速权重丢弃。原loss与原l2在unit weight/all-valid mask下精确一致；模型/Adam严格恢复通过。\n\n|模型|参数|P50秒|P90秒|P99秒|峰值allocated MiB|\n|---|---:|---:|---:|---:|---:|\n'
for r in t['models']:report+=f"|{r['model']}|{r['parameters']}|{r['P50']:.6f}|{r['P90']:.6f}|{r['P99']:.6f}|{r['allocated']/2**20:.1f}|\n"
report+='\n采用P90估计每一步，统一232更新/epoch，三个模型总计139,200次正式更新和17,805,600次目标访问。含I/O、checkpoint、最终验证和20%余量。\n\n|共同epoch|预计总秒|12小时内|\n|---|---:|---|\n'
for r in b['alternatives']:report+=f"|{r['epochs']}|{r['total_projected_seconds']:.1f}|{r['fits']}|\n"
report+=f"\n选定最大共同预算 **{c['epochs']} epoch**。batch128，Adam 0.0001，常数lr，seed42。正式训练前commit配置与执行代码；最终checkpoint比较，禁止挑轮或延长。时长为估算，不保证收敛。GPU账本包括失败预检和恢复，数据准备按日志开始到manifest结束+2秒保守计费{prep:.2f}秒。训练watchdog预留180秒，达到43,020秒停止。\n\n测速原始max_context字段在第2步后记录的是57槽上界；timing_context_verified.csv通过相同目标置换重新核算真实最大人数，后续报告使用此列。此为元数据修正，不涉及样本或训练配置修改。系统python3.8不能载入旧环境numpy1.26扩展的一次报表命令失败，改用隔离Python3.10，不修改旧环境。\n"
(ROOT/'reports/stage9/03_TRAINING_BUDGET.md').write_text(report)
print('Budget sealed before any validation accuracy',c['epochs'])
