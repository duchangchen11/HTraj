"""Paired, complete reporting of the fixed one-recording experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import csv,json,gzip,statistics,subprocess
from collections import defaultdict
import torch
from src.public_training.model import make_model
from src.public_training.model_consistent import make_model as consistent_model
from src.data.preprocessing import sha256,write_json_new
from src.training.social_diagnostics import csv_new

ROOT=Path('results/tables/stage4p1')


def main():
    all_summaries=[];all_curves=[];details={};per_person=[];pairs=[];group_rows=[]
    for kind in ('Full','NoSocial-ConsistentNorm'):
        folder=ROOT/kind;curves=list(csv.DictReader((folder/'curve_30.csv').open()));assert len(curves)==30
        run=json.loads((folder/'execution.json').read_text());smoke=json.loads((folder/'smoke.json').read_text())
        all_curves.extend(curves)
        all_summaries.extend(csv.DictReader((folder/'best_last.csv').open()))
        model=make_model(kind) if kind=='Full' else consistent_model(kind);unused=run['unused_parameter_names'];parameter_map=dict(model.named_parameters())
        details[kind]=dict(best_epoch=run['best_epoch'],best_NLL=run['best_validation_NLL'],parameters=run['parameters'],
            never_called_parameter_count=sum(parameter_map[n].numel() for n in unused),never_called_parameter_names=unused,
            train_seconds=run['training_seconds'],validation_seconds=run['evaluation_seconds'],
            train_windows_per_second=1488*30/run['training_seconds'],
            train_target_windows_per_second=23349*30/run['training_seconds'],
            peak_train_cuda_allocated_MiB=max(float(r['peak_train_cuda_allocated_MiB']) for r in curves),
            initial_train_NLL=float(curves[0]['train_NLL_window_mean']),last_train_NLL=float(curves[-1]['train_NLL_window_mean']),
            minimum_K1_ADE_epoch=min(range(1,31),key=lambda e:float(curves[e-1]['validation_K1_distribution_mean_ADE'])),
            smoke_gate_pass=smoke['gate_pass'],checkpoint_restore_exact=smoke['checkpoint_restore_output_exact'])
    csv_new(ROOT/'model_best_last.csv',all_summaries);csv_new(ROOT/'all_epoch_curves.csv',all_curves)
    banks={}
    for kind in ('Full','NoSocial-ConsistentNorm'):
        with gzip.open(ROOT/kind/'target_errors.csv.gz','rt') as stream:
            banks[kind]={(r['label'],r['mode'],r['window_id'],r['agent_id']):r for r in csv.DictReader(stream)}
    control='NoSocial-ConsistentNorm'
    assert banks['Full'].keys()==banks[control].keys()
    for key,a in banks['Full'].items():
        b=banks[control][key]
        pairs.append(dict(label=key[0],mode=key[1],window_id=key[2],agent_id=key[3],people=int(a['group_people']),
            Full_ADE=float(a['ADE_m']),Full_FDE=float(a['FDE_m']),NoSocial_ADE=float(b['ADE_m']),NoSocial_FDE=float(b['FDE_m']),
            NoSocial_minus_Full_ADE=float(b['ADE_m'])-float(a['ADE_m']),NoSocial_minus_Full_FDE=float(b['FDE_m'])-float(a['FDE_m'])))
    with gzip.open(ROOT/'paired_target_errors.csv.gz','xt',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(pairs[0]));writer.writeheader();writer.writerows(pairs)
    for label in ('best','last'):
        for mode in ('K1_distribution_mean','K1_first_sample','K20_independent_min'):
            rows=[r for r in pairs if r['label']==label and r['mode']==mode]
            for group in ('all','N2to4','N5to8','N9plus'):
                selected=[r for r in rows if group=='all' or (group=='N2to4' and r['people']<=4) or (group=='N5to8' and 5<=r['people']<=8) or (group=='N9plus' and r['people']>=9)]
                row=dict(label=label,mode=mode,group=group,target_windows=len(selected),unique_agents=len({r['agent_id'] for r in selected}))
                for metric in ('Full_ADE','Full_FDE','NoSocial_ADE','NoSocial_FDE','NoSocial_minus_Full_ADE','NoSocial_minus_Full_FDE'):
                    row[metric]=statistics.mean(r[metric] for r in selected) if selected else None
                group_rows.append(row)
            by_person=defaultdict(list)
            for r in rows:by_person[r['agent_id']].append(r)
            for pid,samples in by_person.items():
                per_person.append(dict(label=label,mode=mode,agent_id=pid,window_targets=len(samples),
                    NoSocial_minus_Full_ADE=statistics.mean(r['NoSocial_minus_Full_ADE'] for r in samples),
                    NoSocial_minus_Full_FDE=statistics.mean(r['NoSocial_minus_Full_FDE'] for r in samples)))
    csv_new(ROOT/'paired_context_counts.csv',group_rows);csv_new(ROOT/'paired_person_means.csv',per_person)
    p=json.loads(Path('configs/stage4p1/protocol.json').read_text())
    environment=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,GPU=torch.cuda.get_device_name(),
        prefix=sys.prefix,isolated_stage4_environment=True,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        protocol_sha256=sha256('configs/stage4p1/protocol.json'),manifests=p['manifests'],cache_sha256=sha256(p['cache']),script_sha256=sha256(__file__))
    write_json_new(ROOT/'environment.json',environment)
    write_json_new(ROOT/'analysis.json',dict(models=details,NoSocial_effective_control=control,original_NoSocial_smoke_failed=True,paired_error_rows=len(pairs),unique_validation_agents=len({r['agent_id'] for r in pairs}),
        interpretation='only one exposed recording, seed17; no independent-window significance; positive paired delta means Full lower error; paired_person_means are additional aggregation not independent recordings',
        ETH_evaluated=False,B_trained=False,pretrained_training_initialization=False))
    inventory=[]
    for kind in ('Full','NoSocial','NoSocial-ConsistentNorm'):
        for checkpoint in sorted((Path('experiments/stage4p1')/kind).glob('epoch*.pt')):
            inventory.append(dict(model=kind,path=str(checkpoint),epoch=int(checkpoint.stem[-3:]),bytes=checkpoint.stat().st_size,
                sha256=sha256(checkpoint),failed_pilot=kind=='NoSocial'))
    assert len(inventory)==64
    csv_new(ROOT/'checkpoint_inventory.csv',inventory)
    print(json.dumps(dict(models=details,all_group_best_K1=[r for r in group_rows if r['label']=='best' and r['mode']=='K1_distribution_mean'])))


if __name__=='__main__':main()
