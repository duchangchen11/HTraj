from .common import *
import xml.etree.ElementTree as ET
def main():
    suites=[]
    for f in ['admission_cpu_tests_initial.xml','admission_cpu_tests.xml','admission_cpu_tests_train_only.xml']:
        source=OUT/f
        s=ET.parse(source if source.exists() else gzip.open(OUT/(f+'.gz'),'rb')).getroot().find('testsuite')
        suites.append(dict(file=f,tests=int(s.get('tests')),errors=int(s.get('errors')),failures=int(s.get('failures')),seconds=float(s.get('time'))))
    assert suites[-1]['errors']==suites[-1]['failures']==0 and suites[-1]['tests']==12
    jnew(OUT/'cpu_test_gate.json',dict(passed=True,final_tests=suites[-1],all_attempts=suites,
          wall_seconds_conservative_GPU_budget_charge=sum(s['seconds'] for s in suites),
          initial_failure='Checkpoint metadata helper assumed repository path; temporary pytest path rejected. Fixed path reporting only; all failures retained.',
          model_structure_or_loss_changed=False))
    settings=dict(stage='Stage11B-0',base_commit=BASE,official_commit=OFFICIAL,seed=17,
      real_gpu_training_authorized=True,formal_training_authorized=False,new_model_development=False,
      optimizer='Adam',learning_rate=.0001,loss='Original mean BCE(goal12)+BCE(path12), each x1000',loss_scale=1000,
      scene_batch_size=1,target_batch_size=4,target_batch_source='Explicit official short notebook BATCH_SIZE=4; YAML8 remains recorded alternative, not mixed',
      original_YAML_sha=sha(RES/'config/sdd_trajnet.yaml'),resize=.25,history=8,future=12,waypoints=[11],
      predictor_initialization='Random seed17, no historical predictor checkpoint',
      segmentation_checkpoint_SHA='75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194',
      segmentation_requires_grad=False,segmentation_mode='eval throughout',normalization='Official image preprocessing preserved, including cv2 BGR order',
      smoke_target_batches=[1,4,4],smoke_updates=3,then_reset_same_random_predictor_and_empty_optimizer=True,
      full_passes_max=1,full_pass_augmentation=False,full_pass_targets=7496,full_pass_scenes=25,
      scene_order='torch.randperm(sorted25 scenes), independent generator seed17',within_scene_order='Original author rows, 20 rows per target',
      original_26_scene_cap=False,full_augmentation='CPU count/check only, 8x variants; no additional augmented training run',
      train_only_artifact=json.loads((OUT/'training_data_artifact.json').read_text()),
      inference_after_smoke='Single learned-goal K1 engineering path without GT; no metric',
      resume_probe='Two paired transient updates around strict last-checkpoint restore; do not overwrite last',
      validation_used_for_selection=False,test_pickle_decoded=False,prediction_metrics=False,
      CUDA_deterministic=True,TF32=False,AMP=False,gradient_clipping=None,
      GPU_budget_hard_seconds=3600,safety_stop_conservative_seconds=3300,
      guard_charge='Sum complete GPU process wall + all preflight unit-test wall; later full-suite wall additionally recorded',
      on_OOM='Stop and preserve scene/checkpoint; no automatic batch-size sweep or data removal',
      on_nonfinite='Stop and preserve failure; no retry with changed hyperparameters',
      split_status='SPLIT_IDENTITY_PASS',full_protocol_status='NOT_READY',
      semantic_class_mapping='SEMANTIC_CLASS_MAPPING_UNVERIFIED',semantic_pretraining='SEGMENTATION_EXPOSURE_UNVERIFIED')
    jnew(CONF/'training_admission.json',settings)
    print('All12 pre-GPU CPU gates passed. Configuration fixed; 3 smoke updates and one unaugmented full pass only.')
if __name__=='__main__':main()
