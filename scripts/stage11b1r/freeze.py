"""Write the unique preregistration before any distribution observation."""
from .common import *

def main():
    selected=ROOT/'results/tables/stage11b1/precheck_selected_targets.json'
    cfg=dict(stage='Stage11B-1R',base=BASE,official_commit=OFFICIAL,
        task='SAMPLER_ENGINEERING_ONLY',formal_training=False,new_validation_or_test_metrics=False,
        model='Same randomly initialized seed17 Y-Net core + frozen author segmentation; no predictor checkpoint',
        history=8,future=12,resize=.25,K=20,TTST_proposals=10000,replacement=True,
        relative_threshold=.01,temperature=1.,CWS=False,Ke=20,Ka=1,kmeans_clusters=19,
        kmeans_tolerance=.001,kmeans_iter_limit=1000,
        target_manifest=str(selected.relative_to(ROOT)),target_manifest_SHA=sha(selected),
        CPU_sampler='Official threshold/normalization on original device,float32; copy exact weights to CPU; torch.multinomial with independent Generator',
        proposal_seed="uint32 first4 SHA256('HTraj-YNet-SDD-Development-v1:17:'+sample_id), big endian",
        kmeans_seed="uint32 first4 SHA256('HTraj-YNet-K20-v2:kmeans:'+sample_id), big endian",
        RNG_isolation='Independent CPU proposal Generator; scoped NumPy/CPU/CUDA RNG save+restore for unchanged author KMeans; no repeated global reseeding',
        real_repeat_targets='First3 of same20 training targets,5 full inferences each',
        reverse_order='All20 targets in reversed order',
        independent_process='Repeat first3 targets in fresh interpreter, same model seed/weights',
        future_counterfactual='All20 raw trajectory records, replace only last12 labels by deterministic large constants',
        cache_equivalence='All20 targets: cached segmentation vs image resegmentation',
        exact_reproducibility='Bitwise equality of proposal,goal,path and max coordinate difference0; no post-hoc tolerance',
        distribution_test=dict(maps=['uniform16x20','gradient16x20','sparse16x20','first_real_train_goal_probability'],
            trials=50,draws_per_trial=10000,independent_seed_rule='same stable SHA function on distribution:<map>:trial:<i>',
            spatial_bins='4x4 fixed equal-extent grid in row/column index;16 bins',
            expected='Aggregate original post-threshold GPU-normalized float32 weights into16 bins; normalize in float64 for analysis only',
            method='Simultaneous Hoeffding bound for each empirical bin frequency; union bound over4 maps*2 devices*16 bins',
            familywise_alpha=.001,samples_per_map_device=500000,
            tolerance_formula='epsilon=sqrt(log(2*128/0.001)/(2*500000)); each device-to-expected<=epsilon; CPU-to-GPU<=2*epsilon',
            support='No samples outside nonzero thresholded support; weight tensors CPU vs GPU bitwise equal',
            interpretation='Statistical distribution sanity check, not proof of identical PRNG sequences or theorem of full-model equivalence',
            retry_or_threshold_adjustment=False),
        original_minimal_case=dict(fixed_probability_repeats=5,same_input_full_calls=5,
            fixed_proposal_KMeans_repeats=5,fixed_goal_decoder_repeats=5,
            preserve_probability_tensor_and_proposals=True,official_GPU_control_retained=True),
        GPU_hard_limit_seconds=3600,GPU_safety_stop_seconds=3300,
        GPU_charge='Sum complete worker wall times including interpreter startup measured by parent; CPU tests charged conservatively separately',
        on_failure='Preserve all failure evidence; no formal training, no altered thresholds, no dropping fixed targets')
    jnew(CONF/'sampling_protocol.json',cfg)
    print('Frozen distribution epsilon formula and all sample counts; no errors consulted')

if __name__=='__main__':main()
