"""CPU-only, read-only scoring of frozen predictions; never an inference input.

The primary comparison is fixed epoch30. Best-checkpoint comparison is a
separately labelled secondary development analysis. No missing run is scored.
"""
from pathlib import Path
import csv, gzip, hashlib, json, os, time
import numpy as np
import pandas as pd

from .common import ROOT, OUT, CACHE, CONF, sha, jnew, commit

BASE_OUT = ROOT / 'results/tables/stage11b1c'
BASE_PRED = ROOT / 'experiments/stage11b1c/cache/predictions/epoch030.pt'
BASE_PRED_SHA = '2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab'
BASE_CKPT_SHA = '0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312'
VALID_EPOCHS = (1, 5, 10, 15, 20, 25, 30)
SCENES = ('bookstore_2', 'deathCircle_3', 'gates_7', 'hyang_7', 'nexus_4')
GROUPS = {
    'hist_speed_group': ('low', 'medium', 'high'),
    'hist_turn_group': ('undefined_direction', 'straight', 'intermediate', 'turning_ge30deg'),
    'hist_speed_change_group': ('undefined_low_speed', 'stable_CVle0.15', 'intermediate', 'variable_CVge0.4'),
    'hist_stationary_group': ('all_steps_le1px', 'other'),
}
METRICS = ('ADE1', 'FDE1', 'minADE20', 'minFDE20', 'GoalFDE1', 'minGoalFDE20',
           'GoalCover10', 'GoalCover25', 'GoalCover50', 'GoalPathEndpointMeanDistance',
           'GoalPairwiseDispersion', 'PathPairwiseDispersion', 'SelectionGap_ADE', 'SelectionGap_FDE')
CASE_RULES = dict(semantic_better='Delta_ADE>=1px', no_semantic_better='Delta_ADE<=-1px',
                  close='abs(Delta_ADE)<1px', both_fail='both minGoalFDE20>25px',
                  coverage_changed='GoalCover25 differs', deathCircle='scene==deathCircle_3',
                  other4='scene!=deathCircle_3', selection='lexicographically first sample_id within each eligible cohort and each scene')


def write_csv(path, frame, columns=None):
    """Exclusive creation, LF records; empty results keep an explicit schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(frame) if not isinstance(frame, pd.DataFrame) else frame
    if columns is not None and frame.empty:
        frame = pd.DataFrame(columns=columns)
    with (gzip.open(path, 'xt', newline='') if path.suffix == '.gz' else path.open('x', newline='')) as stream:
        frame.to_csv(stream, index=False, lineterminator='\n')


def metrics_numpy(paths, goals, truth):
    """Independent float64 metrics; ADE and FDE have independent oracle heads."""
    p, g, y = (np.asarray(a, dtype=np.float64) for a in (paths, goals, truth))
    if p.ndim != 4 or p.shape[0] != 20 or p.shape[2:] != (12, 2):
        raise ValueError('Expected K20 full12 trajectories')
    n = p.shape[1]
    if g.shape != (20, n, 2) or y.shape != (n, 12, 2):
        raise ValueError('Prediction/goal/GT shape mismatch')
    if not all(np.isfinite(a).all() for a in (p, g, y)):
        raise ValueError('Nonfinite actual trajectory')
    distance = np.linalg.norm(p-y[None], axis=-1)
    ade, fde = distance.mean(axis=-1), distance[:, :, -1]
    ge = np.linalg.norm(g-y[None, :, -1], axis=-1)
    a, f, q = ade.argmin(0), fde.argmin(0), ge.argmin(0)
    ii = np.arange(n)
    m = dict(ADE1=ade[0], FDE1=fde[0], minADE20=ade[a, ii], minFDE20=fde[f, ii],
             GoalFDE1=ge[0], minGoalFDE20=ge[q, ii], best_ADE_head=a, best_FDE_head=f, best_Goal_head=q,
             GoalPathEndpointMeanDistance=np.linalg.norm(g-p[:, :, -1], axis=-1).mean(0),
             SelectionGap_ADE=ade[0]-ade[a, ii], SelectionGap_FDE=fde[0]-fde[f, ii])
    for r in (10, 25, 50):
        m[f'GoalCover{r}'] = (m['minGoalFDE20'] <= r).astype(float)
    i, j = np.triu_indices(20, 1)
    # Chunking avoids forming a KxKx998x12x2 temporary.
    pdist = np.empty(n); gdist = np.empty(n)
    for first in range(0, n, 64):
        last = min(first+64, n)
        pdist[first:last] = np.linalg.norm(p[i, first:last]-p[j, first:last], axis=-1).mean((0, 2))
        gdist[first:last] = np.linalg.norm(g[i, first:last]-g[j, first:last], axis=-1).mean(0)
    m.update(PathPairwiseDispersion=pdist, GoalPairwiseDispersion=gdist)
    return pd.DataFrame(m), ade, fde, ge


def paired_frame(identities, semantic, no_semantic, comparison, epoch):
    """Exact ID alignment must precede subtraction; positive delta favors semantic."""
    ids = identities.sample_id.tolist()
    if len(ids) != len(set(ids)) or set(semantic.sample_id) != set(ids) or set(no_semantic.sample_id) != set(ids):
        raise ValueError('Missing, duplicate, or extra prediction target ID')
    out = identities.copy().reset_index(drop=True)
    out['comparison'] = comparison; out['NoSemantic_epoch'] = int(epoch); out['Semantic_epoch'] = 30
    for name, data in [('Semantic', semantic), ('NoSemantic', no_semantic)]:
        ordered = data.set_index('sample_id').loc[ids]
        for key in METRICS:
            out[f'{name}_{key}'] = ordered[key].to_numpy()
        for key in ('best_ADE_head', 'best_FDE_head', 'best_Goal_head'):
            out[f'{name}_{key}'] = ordered[key].to_numpy(dtype=int)
    out['Delta_ADE'] = out.NoSemantic_minADE20-out.Semantic_minADE20
    out['Delta_FDE'] = out.NoSemantic_minFDE20-out.Semantic_minFDE20
    out['Delta_GoalCover'] = out.Semantic_GoalCover25-out.NoSemantic_GoalCover25
    out['Semantic_A25_failed'] = out.Semantic_GoalCover25 == 0
    out['NoSemantic_A25_failed'] = out.NoSemantic_GoalCover25 == 0
    return out


def scope_frames(frame):
    yield 'ALL998', frame
    for scene in SCENES:
        yield scene, frame[frame.scene == scene]
    yield 'OTHER4', frame[frame.scene != 'deathCircle_3']


def summarize(frame, scope, weighting='target_window'):
    if not len(frame):
        return dict(scope=scope, weighting=weighting, targets=0, people=0, recordings=0, status='EMPTY_GROUP')
    numeric = [f'{name}_{key}' for name in ('Semantic', 'NoSemantic') for key in METRICS]
    numeric += ['Delta_ADE', 'Delta_FDE', 'Delta_GoalCover']
    if weighting == 'target_window':
        mean = frame[numeric].mean()
    elif weighting == 'person_ID_equal':
        mean = frame.groupby(['scene', 'person_id'], sort=True)[numeric].mean().mean()
    elif weighting == 'recording_equal':
        mean = frame.groupby('scene', sort=True)[numeric].mean().mean()
    elif weighting == 'recording_equal_person_ID_equal':
        mean = frame.groupby(['scene', 'person_id'], sort=True)[numeric].mean().groupby(level='scene').mean().mean()
    else:
        raise ValueError('Unknown weighting')
    row = dict(scope=scope, weighting=weighting, targets=len(frame), people=len(frame[['scene', 'person_id']].drop_duplicates()),
               recordings=frame.scene.nunique(), status='COMPLETE', **{k: float(v) for k, v in mean.items()})
    row.update(Semantic_A25_failed=int(frame.Semantic_A25_failed.sum()), NoSemantic_A25_failed=int(frame.NoSemantic_A25_failed.sum()),
               A25_failure_count_change=int(frame.NoSemantic_A25_failed.sum()-frame.Semantic_A25_failed.sum()),
               semantic_lower_ADE_fraction=float((frame.Delta_ADE>0).mean()),
               no_semantic_lower_ADE_fraction=float((frame.Delta_ADE<0).mean()),
               identical_ADE_fraction=float((frame.Delta_ADE==0).mean()),
               scene_composition=json.dumps(frame.scene.value_counts().sort_index().to_dict(), sort_keys=True))
    return row


def comparison_summaries(frame):
    rows = []
    for scope, sub in scope_frames(frame):
        for weight in ('target_window', 'person_ID_equal'):
            rows.append(summarize(sub, scope, weight))
    rows.append(summarize(frame, 'FIVE_RECORDING_MACRO', 'recording_equal'))
    rows.append(summarize(frame, 'FIVE_RECORDING_MACRO', 'recording_equal_person_ID_equal'))
    for r in rows:
        r.update(comparison=frame.comparison.iloc[0], Semantic_epoch=30, NoSemantic_epoch=int(frame.NoSemantic_epoch.iloc[0]))
    return rows


def semantic_gate(frame):
    """Frozen single-seed screening: denominator is always NoSemantic metric."""
    macro = summarize(frame, 'FIVE_RECORDING_MACRO', 'recording_equal')
    other = summarize(frame[frame.scene != 'deathCircle_3'], 'OTHER4', 'target_window')
    if frame.scene.nunique() != 5 or not len(frame[frame.scene != 'deathCircle_3']):
        return dict(status='NOT_EVALUABLE', reason='All five recordings required')
    gain = (macro['NoSemantic_minADE20']-macro['Semantic_minADE20'])/macro['NoSemantic_minADE20']
    fde_degrade = (macro['Semantic_minFDE20']-macro['NoSemantic_minFDE20'])/macro['NoSemantic_minFDE20']
    other_degrade = (other['Semantic_minADE20']-other['NoSemantic_minADE20'])/other['NoSemantic_minADE20']
    count = int(sum(group.Delta_ADE.mean()>0 for _, group in frame.groupby('scene')))
    conditions = dict(macro_ADE_semantic_gain_ge2pct=bool(gain>=.02), macro_FDE_semantic_degradation_le1pct=bool(fde_degrade<=.01),
                      at_least3_of5_recordings_favor_semantic=bool(count>=3), other4_pooled_semantic_ADE_degradation_le1pct=bool(other_degrade<=.01))
    return dict(status='SEMANTIC_BENEFIT_PRELIMINARY_GO' if all(conditions.values()) else 'SEMANTIC_BENEFIT_INCONCLUSIVE_OR_NO_GO',
                conditions=conditions, failed_conditions=[k for k, v in conditions.items() if not v],
                measurements=dict(macro_ADE_relative_semantic_gain=gain, macro_FDE_relative_semantic_degradation=fde_degrade,
                                  recordings_favoring_semantic=count, other4_relative_semantic_ADE_degradation=other_degrade),
                denominator='NoSemantic metric; OTHER4 is target-window pooled',
                interpretation='Single-seed repeatedly exposed development screening; not significance or final causal proof')


def a25_rows(frame):
    rows = []
    definitions = {
        'both_failed': frame.Semantic_A25_failed & frame.NoSemantic_A25_failed,
        'semantic_only_failed_recovered_by_NoSemantic': frame.Semantic_A25_failed & ~frame.NoSemantic_A25_failed,
        'NoSemantic_only_failed_newly_lost': ~frame.Semantic_A25_failed & frame.NoSemantic_A25_failed,
        'both_successful': ~frame.Semantic_A25_failed & ~frame.NoSemantic_A25_failed,
        'original_semantic_164_failures': frame.Semantic_A25_failed,
        'original_semantic_834_successes': ~frame.Semantic_A25_failed,
    }
    for scope, subset in scope_frames(frame):
        for cohort, mask in definitions.items():
            group = frame.loc[mask & frame.index.isin(subset.index)]
            row = summarize(group, scope)
            row.update(comparison=frame.comparison.iloc[0], cohort=cohort,
                       sample_ids_SHA=hashlib.sha256('\n'.join(sorted(group.sample_id)).encode()).hexdigest())
            rows.append(row)
    return rows


def choose_cases(frame):
    masks = dict(semantic_better=frame.Delta_ADE>=1, no_semantic_better=frame.Delta_ADE <= -1,
                 close=frame.Delta_ADE.abs()<1, both_fail=frame.Semantic_A25_failed & frame.NoSemantic_A25_failed,
                 coverage_changed=frame.Semantic_GoalCover25 != frame.NoSemantic_GoalCover25,
                 deathCircle=frame.scene=='deathCircle_3', other4=frame.scene!='deathCircle_3')
    masks.update({scene: frame.scene==scene for scene in SCENES})
    rows = []
    for label, mask in masks.items():
        sub = frame[mask].copy()
        if not len(sub):
            rows.append(dict(cohort=label, status='NO_ELIGIBLE_CASE', eligible=0, sample_id=None, scene=None)); continue
        r = sub.sort_values('sample_id').iloc[0]
        rows.append(dict(cohort=label, status='SELECTED', eligible=len(sub), sample_id=r.sample_id, scene=r.scene,
                         Semantic_minADE20=float(r.Semantic_minADE20),
                         NoSemantic_minADE20=float(r.NoSemantic_minADE20), Delta_ADE=float(r.Delta_ADE)))
    return rows


def _load_pack(path, expected_sha, ids):
    import torch
    if expected_sha is not None and sha(path) != expected_sha:
        raise ValueError('Frozen prediction SHA mismatch')
    pack = torch.load(path, map_location='cpu', weights_only=True)
    if pack['K'] != 20 or pack['TTST_proposals'] != 10000 or set(pack['sample_ids']) != set(ids):
        raise ValueError('Prediction identity or candidate-budget mismatch')
    if len(pack['sample_ids']) != len(set(pack['sample_ids'])):
        raise ValueError('Duplicate target prediction')
    index = {sid: i for i, sid in enumerate(pack['sample_ids'])}
    order = [index[sid] for sid in ids]
    return pack, pack['paths'][:, order].numpy(), pack['goals'][:, order].numpy()


def write_unrun_analysis(execution):
    """Budget rejection provides no new prediction evidence and reads no GT."""
    paths = {
        'paired_comparison.csv': ['comparison','status','sample_id','scene','history_SHA','GT_SHA',
                                  'Semantic_minADE20','NoSemantic_minADE20','Delta_ADE'],
        'scene_comparison.csv': ['comparison','scope','weighting','status','targets',
                                 'Semantic_minADE20','NoSemantic_minADE20','Delta_ADE'],
        'motion_group_comparison.csv': ['comparison','scope','factor','group','status','targets',
                                        'Semantic_minADE20','NoSemantic_minADE20'],
        'a25_failure_comparison.csv': ['comparison','scope','cohort','status','targets'],
        'fixed_epoch30_comparison.csv': ['comparison','status','reason'],
        'best_checkpoint_comparison.csv': ['comparison','status','reason'],
        'leave_one_recording_out.csv': ['comparison','excluded_recording','status'],
        'candidate_paired_errors.csv.gz': ['epoch','sample_id','head','ADE','FDE','GoalFDE'],
        'qualitative_case_manifest.csv': ['cohort','status','sample_id','scene'],
    }
    for name, columns in paths.items():
        write_csv(OUT/name, [], columns)
    source = BASE_OUT/'paired_epoch030.csv'
    old = pd.read_csv(source)
    if len(old) != 998 or old.sample_id.duplicated().any() or int((old.minGoalFDE20>25).sum()) != 164:
        raise ValueError('Existing semantic reference integrity mismatch')
    no_evidence = dict(status='NOT_EVALUABLE', reason='Formal NoSemantic training and full validation not started: budget admission rejected')
    result = dict(stage='Stage11C1', scope='MATCHED_TRAINING_CONTROL_ONLY',
        status='BUDGET_NO_GO_SEMANTIC_BENEFIT_NOT_EVALUATED',
        engineering_status='PREFLIGHT_ONLY', budget_status='NO_GO',
        semantic_benefit_status='NOT_EVALUATED', empirical_semantic_failure_claim=False,
        primary_comparison='FIXED_EPOCH30', secondary_comparison='BEST_COMPLETE_VALIDATION',
        primary_research_endpoint='FIVE_RECORDING_MACRO minADE20',
        primary_gate=no_evidence, secondary_gate=no_evidence,
        complete_validation_epochs=[], expected_validation_epochs=list(VALID_EPOCHS),
        full_seven_validations_completed=False, formal_epochs_completed=0,
        baseline_A25_failures=164, baseline_A25_successes=834,
        baseline_only_source=str(source.relative_to(ROOT)), baseline_source_SHA=sha(source),
        comparison_status=dict(FIXED_EPOCH30='NOT_RUN', BEST_COMPLETE_VALIDATION='NOT_RUN'),
        delta_sign='NoSemantic minus Semantic for ADE/FDE; Semantic minus NoSemantic for coverage; positive favors Semantic',
        empty_result_tables='Header-only; no NoSemantic metrics, deltas, transition counts, best epoch, or research gates imputed',
        limitations=['Conservative full-run estimate uses a small untrained engineering probe, not a measured full30-epoch runtime.',
            'No formal NoSemantic training or998-target validation exists, so semantic benefit cannot be accepted or rejected empirically.',
            'Same nominal model does not guarantee equal effective input capacity because six semantic input channels are zero.',
            'One seed and five reused development recordings would not constitute independent confirmation.',
            'Existing semantic reference values are frozen historical results, not results of this stage.'],
        semantic_quality_causal_claim='NOT_SUPPORTED', gating_module='NOT_AUTHORIZED_OR_VALIDATED',
        multi_seed_recommendation='NOT_SUPPORTED_WITHOUT_COMPLETED_MATCHED_CONTROL',
        automatic_additional_training=False,
        next_question='Can a user-approved cumulative GPU budget accommodate the unchanged matched30-epoch run and all seven full validations?',
        thesis_direction_change='No empirical basis from this unexecuted control', case_selection_rules=CASE_RULES,
        formal_training_execution_status=execution.get('status'))
    jnew(OUT/'final_decision.json', result)
    jnew(OUT/'analysis_manifest.json', dict(status='COMPLETE_NO_NEW_PREDICTIONS', executed_commit=commit(),
        CPU_only=True, future_GT_loaded=False, test_loaded=False, metrics_calculated=False,
        all_new_comparisons_unrun=True, baseline_A25_count_verified_from_frozen_old_table=True,
        baseline_source=dict(path=str(source.relative_to(ROOT)),sha256=sha(source)),
        code_SHA=sha(Path(__file__)), no_missing_metrics_imputed=True,
        empty_tables=list(paths), result_status=result['status']))
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    start = time.perf_counter()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('Run offline analysis with CUDA_VISIBLE_DEVICES empty')
    execution_path = OUT/'training_execution.json'
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
        if int(execution.get('completed_epochs', 0)) == 0 and not execution.get('completed_validation_epochs', []):
            write_unrun_analysis(execution)
            return
    from scripts.stage11b1c.data import manifest, truths
    identities = pd.DataFrame(manifest('validation'))
    features_path = ROOT/'results/tables/stage11b2a/motion_features.csv'
    features = pd.read_csv(features_path)
    columns = ['sample_id', *GROUPS]
    for key in ('history_SHA', 'GT_SHA', 'scene', 'person_id', 'frame_id'):
        left = identities.set_index('sample_id')[key].astype(str)
        right = features.set_index('sample_id')[key].astype(str)
        if set(left.index) != set(right.index) or not left.sort_index().equals(right.sort_index()):
            raise ValueError('Frozen identity/history/GT differs from Stage11B2A: '+key)
    identities = identities.merge(features[columns], on='sample_id', validate='one_to_one')
    ids = identities.sample_id.tolist()
    if len(ids) != 998 or set(identities.scene) != set(SCENES):
        raise ValueError('Frozen998 five-recording validation required')
    ymap = truths('validation'); y = np.stack([ymap[s] for s in ids])
    bp, paths, goals = _load_pack(BASE_PRED, BASE_PRED_SHA, ids)
    if bp['checkpoint_SHA'] != BASE_CKPT_SHA:
        raise ValueError('Semantic checkpoint identity differs')
    semantic, _, _, _ = metrics_numpy(paths, goals, y)
    semantic['sample_id'] = ids
    old = pd.read_csv(BASE_OUT/'paired_epoch030.csv').set_index('sample_id').loc[ids]
    primary_discrepancy = max(float(np.max(np.abs(semantic[k].to_numpy()-old[k].to_numpy())))
                              for k in ('ADE1', 'FDE1', 'minADE20', 'minFDE20'))
    if primary_discrepancy > 1e-6:
        raise ValueError('Independent primary scoring disagrees with semantic reference')
    if int((semantic.minGoalFDE20>25).sum()) != 164:
        raise ValueError('Original164 failure cohort changed')
    results = []
    for epoch in VALID_EPOCHS:
        p = OUT/f'validation_epoch{epoch:03d}.json'
        if p.exists():
            r = json.loads(p.read_text())
            if r.get('status') == 'COMPLETE' and r.get('targets_scored') == 998:
                results.append(r)
    best = min(results, key=lambda r: (r['window']['minADE20'], r['epoch'])) if results else None
    fixed = next((r for r in results if r['epoch']==30), None)
    comparisons = [('FIXED_EPOCH30', fixed), ('BEST_COMPLETE_VALIDATION', best)]
    all_paired, scenes, groups, failures, loo, candidate_rows = [], [], [], [], [], []
    statuses, gates, selected_cases = {}, {}, []
    scored = {}
    for label, result in comparisons:
        name = 'fixed_epoch30_comparison.csv' if label=='FIXED_EPOCH30' else 'best_checkpoint_comparison.csv'
        if result is None:
            statuses[label] = 'NOT_RUN_OR_INCOMPLETE'
            write_csv(OUT/name, [dict(comparison=label, status='NOT_RUN_OR_INCOMPLETE', reason='No complete prediction cache for required checkpoint')])
            gates[label] = dict(status='NOT_EVALUABLE', reason='Required complete validation unavailable')
            continue
        epoch = int(result['epoch']); info = result['prediction_manifest']; path = ROOT/info['path']
        pack, pp, gg = _load_pack(path, info['sha256'], ids)
        if pack['checkpoint_SHA'] != result['checkpoint']['sha256']:
            raise ValueError('Checkpoint/prediction mismatch')
        if epoch not in scored:
            m, ade, fde, ge = metrics_numpy(pp, gg, y); m['sample_id'] = ids
            for key in ('ADE1', 'FDE1', 'minADE20', 'minFDE20'):
                if abs(float(m[key].mean())-float(result['window'][key])) > 1e-6:
                    raise ValueError('Independent scoring differs from validation: '+key)
            scored[epoch] = m
            for i, sid in enumerate(ids):
                for head in range(20):
                    candidate_rows.append(dict(epoch=epoch, sample_id=sid, head=head, ADE=ade[head,i], FDE=fde[head,i], GoalFDE=ge[head,i]))
        paired = paired_frame(identities, semantic, scored[epoch], label, epoch)
        paired['Semantic_checkpoint_SHA'] = BASE_CKPT_SHA
        paired['NoSemantic_checkpoint_SHA'] = result['checkpoint']['sha256']
        all_paired.append(paired); summary = comparison_summaries(paired)
        scenes.extend(summary); write_csv(OUT/name, summary); failures.extend(a25_rows(paired))
        for factor, levels in GROUPS.items():
            for scope, sub in scope_frames(paired):
                for level in levels:
                    group = sub[sub[factor]==level]
                    for weighting in ('target_window', 'person_ID_equal'):
                        row = summarize(group, scope, weighting)
                        row.update(comparison=label, factor=factor, group=level, grouping_scope='HISTORY_INPUT_ONLY',
                                   scene_composition=json.dumps(group.scene.value_counts().sort_index().to_dict(),sort_keys=True))
                        groups.append(row)
        for excluded in ['NONE', *SCENES]:
            sub = paired if excluded=='NONE' else paired[paired.scene!=excluded]
            row = summarize(sub, 'REMAINING_RECORDING_MACRO', 'recording_equal')
            row.update(comparison=label, excluded_recording=excluded, interpretation='Descriptive sensitivity; five reused recording clusters, overlapping windows')
            loo.append(row)
        gates[label] = semantic_gate(paired); statuses[label] = 'COMPLETE'
        if label=='FIXED_EPOCH30' or fixed is None:
            selected_cases = choose_cases(paired)
            for row in selected_cases:
                row.update(comparison=label, NoSemantic_epoch=epoch)
    empty_columns = ['comparison', 'status', 'sample_id', 'scene', 'history_SHA', 'GT_SHA']
    write_csv(OUT/'paired_comparison.csv', pd.concat(all_paired, ignore_index=True) if all_paired else [], empty_columns)
    write_csv(OUT/'scene_comparison.csv', scenes, ['comparison','scope','weighting','status','targets'])
    write_csv(OUT/'motion_group_comparison.csv', groups, ['comparison','scope','factor','group','status','targets'])
    write_csv(OUT/'a25_failure_comparison.csv', failures, ['comparison','scope','cohort','status','targets'])
    write_csv(OUT/'leave_one_recording_out.csv', loo, ['comparison','excluded_recording','status'])
    write_csv(OUT/'candidate_paired_errors.csv.gz', candidate_rows, ['epoch','sample_id','head','ADE','FDE','GoalFDE'])
    write_csv(OUT/'qualitative_case_manifest.csv', selected_cases, ['cohort','status','sample_id','scene'])
    complete30 = fixed is not None
    decision = dict(stage='Stage11C1', scope='MATCHED_TRAINING_CONTROL_ONLY', comparison_status=statuses,
                    primary_comparison='FIXED_EPOCH30', secondary_comparison='BEST_COMPLETE_VALIDATION',
                    primary_research_endpoint='FIVE_RECORDING_MACRO minADE20',
                    checkpoint_selection='minimum target-window validation minADE20, earliest exact tie',
                    primary_gate=gates['FIXED_EPOCH30'], secondary_gate=gates['BEST_COMPLETE_VALIDATION'],
                    status=gates['FIXED_EPOCH30']['status'] if complete30 else 'INCOMPLETE_MATCHED_CONTROL_NO_RESEARCH_GO',
                    complete_validation_epochs=[r['epoch'] for r in results], expected_validation_epochs=list(VALID_EPOCHS),
                    full_seven_validations_completed=[r['epoch'] for r in results]==list(VALID_EPOCHS),
                    baseline_A25_failures=164, baseline_A25_successes=834,
                    delta_sign='NoSemantic minus Semantic for ADE/FDE; Semantic minus NoSemantic for coverage; positive favors Semantic',
                    limitations=['One seed, five repeatedly used development recordings; windows are not independent.',
                        'Six zero input channels reduce effective input connections despite matching nominal architecture.',
                        'Ego trajectory coordinates and scene image-grid dimensions remain available.',
                        'This tests training-time semantic content, not segmentation correctness, calibration, or semantic gating.',
                        'minADE20/minFDE20 select independent oracle candidates; K1 is not replaced by K20.'],
                    semantic_quality_causal_claim='NOT_SUPPORTED', gating_module='NOT_AUTHORIZED_OR_VALIDATED',
                    automatic_additional_training=False,
                    next_question='Only if the frozen primary gates pass: does the same matched semantic-content effect repeat across prespecified seeds?',
                    case_selection_rules=CASE_RULES)
    jnew(OUT/'final_decision.json', decision)
    inputs = [BASE_PRED, BASE_OUT/'paired_epoch030.csv', ROOT/'results/tables/stage11b2a/motion_features.csv',
              BASE_OUT/'validation_target_manifest.csv']
    for result in results:
        inputs.extend([OUT/f"validation_epoch{result['epoch']:03d}.json", ROOT/result['prediction_manifest']['path']])
    jnew(OUT/'analysis_manifest.json', dict(status='COMPLETE', executed_commit=commit(), seconds=time.perf_counter()-start,
          CPU_only=True, test_loaded=False, independent_primary_max_abs_difference=primary_discrepancy,
          reference_primary_tolerance=1e-6, complete_comparisons=statuses, case_selection_rules=CASE_RULES,
          inputs=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in inputs],
          code_SHA=sha(Path(__file__)), no_missing_metrics_imputed=True))
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
