"""Identity-based source-train development proposal, without calculating errors."""
from common import *
def main():
    a=json.loads((OUT/'split_identity_audit.json').read_text())
    pipe=json.loads((OUT/'full_pipeline_admission.json').read_text())
    maps=json.loads((OUT/'semantic_map_audit.json').read_text())
    candidate=dict(name='HTraj-YNet-SDD-source-train-recording-dev-v1',status='NOT_READY',formal_training_authorized=False,
        source_commit=MODEL_COMMIT,author_archive_SHA=json.loads((OUT/'download_manifest.json').read_text())['sha256'],
        draft_reason='Numerical identity/time separation passes; semantic ID/name and checkpoint segmentation pretraining exposure remain unverified. No source-train validation in author bundle.',
        base_task=dict(history=8,future=12,original_fps=30,original_fps_evidence='Official preprocessing docstring; original videos not measured',sampled_frame_step=12,sampled_dt_seconds=.4,
                       observed_first_to_last_seconds=2.8,future_last_obs_to_endpoint_seconds=4.8,total_first_to_last_seconds=7.6,
                       observed_unit='raw reference pixels',world_scale=None,meters_claimed=False),
        author_protocol=dict(train_videos=a['author_train']['scenes'],test_videos=a['author_test']['scenes'],validation=None,
                             train_targets=a['author_train']['targets'],test_targets=a['author_test']['targets'],test_as_validation_reproduction_forbidden=True,
                             training_notebook_batch_size=4,yaml_batch_size=8,batch_size_conflict='must explicitly freeze entry point, not silently mix',
                             native_epoch_cap='26 scene batches in short-term train.py; after8-way augmentation not a full traversal'),
        proposed_protocol=dict(validation_rule="For each physical site with >=2 author-train recordings, take min SHA256('HTraj-SDD-dev-v1:'+sceneId); singleton sites remain train",
                               val_videos=a['candidate_val_scenes'],train_videos=sorted(set(a['author_train']['scenes'])-set(a['candidate_val_scenes'])),
                               test_videos=a['author_test']['scenes'],counts=a['candidate_counts'],
                               sampled_based_on_validation_errors=False,from_scratch_prediction_training_required=True,
                               author_pretrained_predictor_valid_for_new_dev_generalization=False,
                               pretrained_predictor_reason='It may have trained on proposed validation recordings and selected using official test. Engineering forward only, no independent dev estimates.'),
        identity=dict(key='sceneId + trackId + 20 frame support',manifest=str((OUT/'development_target_manifest_candidate.csv').relative_to(ROOT)),manifest_SHA=sha(OUT/'development_target_manifest_candidate.csv'),
                      complete_20_each=True,within_window_step12=True,same_scene_track_has_one_20_frame_segment=True,
                      stride='Not identifiable: one segment per scene-track; do not invent sliding stride20',
                      original_track_and_window_preprocessing_provenance='PARTIAL: original data generation list/raw annotations absent',
                      out_of_image_train_points=a['author_train']['outside_reference_points'],out_of_image_policy='retain; no silent clipping/dropping; examine author template/padding at later training admission'),
        contextual_scope='YNet has individual trajectory heatmaps + static scene; no explicit other-agent tokens; do not add future-qualified neighbors',
        flags=dict(lost='Absent from processed pickle; official loader removes lost; per-file raw equality not established',occluded='Absent from processed pickle; raw official loader does not discard',generated='Absent from processed pickle; raw official loader retains',target_class='Author SDD pedestrian task; no class field left in processed pool; cannot independently reclassify each target'),
        overlap=dict(exact_window_intersection=a['exact_window_intersection'],recording_intersection=[],
                     shared_static_background_sites=a['site_background_overlap'],not_unseen_site_generalization=True,
                     numeric_track_ID_collision_across_videos_not_identity=True,possible_same_person_cross_camera_unverified=True),
        semantics=dict(author_masks=maps['total_maps'],IDs_present=[1,2,3,4,5],checkpoint_channels=6,channel_0_name='UNVERIFIED; no0pixels in supplied SDD maps',
                       SDD_id_name_mapping='UNVERIFIED; inD README mapping applies to inD, only a clue for SDD',
                       segmentation_pretraining_split='UNVERIFIED; all60 masks include17test recordings',
                       exact_raw_video_registration='UNVERIFIED; no original videos or reference frame IDs',training_input='Predicted6-channel softmax from RGB; GT masks not substituted'),
        evaluation_future_design=dict(K1_and_K20_separate=True,independent_minADE_and_minFDE_full_paths=True,official_goalFDE_separate=True,
                    no_test_metrics_this_stage=True,no_test_checkpoint_selection=True,GT_never_prediction_input=True),
        engineering=dict(strict_load=pipe['strict_load'],real_pipeline_pass=pipe['passed'],does_not_freeze_protocol=True),
        readiness_blockers=['SDD per-class ID/name/ignore mapping lacks SDD-specific authority',
                            'Semantic segmentation checkpoint training/validation/test label exposure unknown',
                            'Original short pool selection/stride provenance incomplete',
                            'Proposed source-train-only validation and training schedule require review; official notebook uses test',
                            'Current Torch2.5 compatibility is functional; numerical equivalence to Torch1.5 not established'])
    jnew(CONF/'protocol_candidate.json',candidate)
    print('Protocol NOT_READY; separated',a['candidate_counts'])
if __name__=='__main__':main()
