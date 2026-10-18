from common import *


def main():
    data = json.loads((OUT / 'data_audit.json').read_text())
    write_json(CONF / 'protocol.json', dict(
        stage='10A', base_branch='feat/stage9-resonance-pairing-controlled-pilot',
        base_commit='d8e753f718647db49202da3ac6d43e4b37cfc59b',
        branch='feat/stage10a-riskgraph-feasibility',
        data_protocol='HTraj-RiskGraph-Causal8x12-v1',
        target_pool_reference='HTraj-SocialVAE-Causal8x12-v1', manifests=data['manifests'],
        history=8, prediction=12, dt=.4, unit='m', neural_output='deterministic K1',
        validation=['biwi_hotel','crowds_zara02'], ETH_sealed=True,
        independent_confirmation=False, formal_training=False, new_ADE_FDE=False,
        graph_config=dict(path='configs/stage10a/graph.json', sha256=sha(CONF / 'graph.json')),
        motion='backward continuous differences only; explicit known flag; zero placeholder for unknown',
        visibility='current co-visible sender/receiver; unknown velocity allowed as position-only message with explicit flag',
        S='pair first co-visible distance<=radius; nearest/stableID among new pairs fills permanent K slots; edges thereafter visibility-masked only; rejected pairs never reconsidered',
        D='current co-visible within radius, lowest current distance topK, stable source-ID ties',
        R='exact D candidate pool and K, highest risk topK then stable sourceID; missing velocity distance-only risk fallback',
        shared_message_features='relative position2, relative velocity2, receiver/source velocity-known2, distance1,DCA1,tCPA1',
        model=dict(hidden_dim=64, encoder='GRUCell(3,64), per-agent causal reset on missing/reentry',
            messages='single-hop MLP(73,64,64), ReLU; mean over incoming edges, then mean over message-bearing continuous history frames',
            decoder='MLP(128,64,24), ReLU; 12 cumulative displacements anchored at last historical position',
            normalization=None, gate=None, high_order=False, future_loss_reference='masked mean Euclidean regression, not executed'),
        no_neighbor='social vector exactly zero; identical weights E vs social returns identical prediction',
        capacity='same nominal parameters all four; E message and decoder social columns inactive; E not sufficient sole capacity control',
        synthetic_smoke=dict(seed=17,batch_scenes=8,people=[1,12,32,79],warmup=2,timed_repeats=8,
            optimizer_updates=0, objective='synthetic output square mean for backward only', device='CPU and RTX3080 CUDA'),
        estimated_future_budget=dict(authorized=False,models=['E','S','D','R'],folds=2,seed=17,epochs=30,
            scene_batch=8,steps='ceil(supervised_scene_windows/8)*30; all target labels each epoch',
            hypothetical_optimizer='Adam lr.001 clip1, no scheduler; requires separate Stage10B freeze',
            checkpoint_rule='lowest validation target-window ADE, earliest exact ties; paired FDE',
            decision='no Stage10B automatic launch; novelty overlap must be resolved or scoped as replication'),
        refusal_conditions=['nonfinite output/gradient','future leakage','permutation failure','R/D always identical',
            'unbounded memory or OOM; no deleting crowded scenes to pass'],
        reuse=dict(code='Original Stage10 implementation; only read-only HTraj Recording/history_context and SHA utilities interfaces reused',
            third_party_frozen_source_edited=False, model_checkpoint_loaded=False,
            license_note='Local parsers authored within HTraj; no third-party model code copied; vendor licenses remain unchanged'),
    ))
    write_json(OUT / 'initial_source_manifest.json', dict(execution_commit=commit(),
        files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for base in ['src/risk_dynamic_graph','scripts/stage10a','configs/stage10a']
            for p in sorted((ROOT/base).rglob('*')) if p.is_file() and '__pycache__' not in p.parts],
        command='No training or validation metrics; engineering only'))


if __name__ == '__main__':
    main()
