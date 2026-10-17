# CVAE完整epoch成本测量

实际执行状态：success=True。

```json
{
  "P50_seconds": 0.1535781740094535,
  "P90_seconds": 0.21016074482467959,
  "P99_seconds": 0.2992931450012838,
  "RNG_saved": true,
  "all_step_mean_seconds": 0.16232855578755598,
  "author_initialization": false,
  "candidate_frame_slots": 4349,
  "checkpoint": "/home/lrj/HTraj/experiments/stage7p4/cache/execution_v2/training_workspace/results/hotel_agentformer_pre/models/model_0001.p",
  "checkpoint_SHA256": "7d770983ff99516edc600d73c771b54c1c9965fbd2a66e01927257071dce39e5",
  "checkpoint_bytes": 72472184,
  "checkpoint_save_seconds": 0.24292708898428828,
  "context_truncated": false,
  "external_CVAE_state_before": null,
  "external_CVAE_unchanged": null,
  "final_state": {
    "parameter_L2": 302.0719377742081,
    "parameter_count": 5972098,
    "parameter_max_abs": 1.0378286838531494,
    "parameter_state_sha256": "9287cee08accd91b9a25ebb02ad631e785610c8d053832ff65079557939871e3",
    "parameters_finite": true
  },
  "full_epochs_completed": 1,
  "full_target_windows": 29676,
  "initial_state": {
    "parameter_L2": 301.8160469556727,
    "parameter_count": 5972098,
    "parameter_max_abs": 1.0,
    "parameter_state_sha256": "fc0322e31988be416e58c959950a1eef7518289918a7fbbf6fdb71a71f26bc8d",
    "parameters_finite": true
  },
  "learning_rate": 0.0001,
  "loader_initialization_seconds": 0.7916413219645619,
  "max_context_people": 57,
  "official_configured_epochs": 100,
  "official_training_source_SHA256": "5774ccaca4043b67e80fa35a5651be7397a6a0fc1de4e2d61dc6c92cdfff4853",
  "optimizer_updates": 3118,
  "original_config": "hotel_agentformer_pre",
  "random_initialization": true,
  "raw_log": "/home/lrj/HTraj/experiments/stage7p4/cache/execution_v2/CVAE/official_train_stdout.txt",
  "raw_log_SHA256": "d6703d4fc3d107ab8559e8b033ebc63cdea13f1efc2a7bfeb150a85280e3a622",
  "resource": {
    "OOM": false,
    "peak_allocated_bytes": 6471955968,
    "peak_reserved_bytes": 9590276096,
    "seconds": 509.88280144898454,
    "whole_card_memory_samples": 3745,
    "whole_card_sampled_peak_MiB": 9677,
    "whole_card_sampling_interval_seconds": 0.1
  },
  "resume_probe": {
    "RNG_exact": true,
    "base_CVAE_unchanged": null,
    "finite_loss": 18.402385711669922,
    "frame": 128,
    "gradient_norm": 280.32598876953125,
    "model_state_after": {
      "parameter_L2": 302.07200382646477,
      "parameter_count": 5972098,
      "parameter_max_abs": 1.0378201007843018,
      "parameter_state_sha256": "e7b5ce82065958cfc8eebaeee80f977e3f628805b76dd6e4a66c70de78c9927c",
      "parameters_finite": true
    },
    "model_state_before": {
      "parameter_L2": 302.0719377742081,
      "parameter_count": 5972098,
      "parameter_max_abs": 1.0378286838531494,
      "parameter_state_sha256": "9287cee08accd91b9a25ebb02ad631e785610c8d053832ff65079557939871e3",
      "parameters_finite": true
    },
    "not_an_extra_epoch": true,
    "optimizer_entries": 134,
    "optimizer_state_exact": true,
    "optimizer_step_max_after": 3119.0,
    "optimizer_step_max_before": 3118.0,
    "optimizer_step_min_after": 3119.0,
    "optimizer_step_min_before": 3118.0,
    "optimizer_update_probe": 1,
    "people": 4,
    "scheduler_state_exact": true,
    "seconds": 1.3372829540167004,
    "sequence": "crowds_zara02_train",
    "strict_model_restore": true
  },
  "scheduler_state": {
    "_get_lr_called_within_step": false,
    "_last_lr": [
      0.0001
    ],
    "_step_count": 2,
    "base_lrs": [
      0.0001
    ],
    "gamma": 0.5,
    "last_epoch": 1,
    "step_size": 10,
    "verbose": false
  },
  "seed": 1,
  "stage": "CVAE",
  "stage_execution_epochs": 1,
  "steady_mean_seconds": 0.16217370439782494,
  "steady_step_count": 3108,
  "steady_warmup_updates_excluded": 10,
  "success": true,
  "tensorboard_scalar_calls": 828,
  "total_data_read_seconds": 34.40221096004825,
  "total_gradient_and_parameter_check_seconds": 13.792112332826946,
  "updates": 3118,
  "valid_windows": 3118
}
```

每次有效窗口的原始loss分项、梯度范数、上下文人数、同步step时间及显存见同名_steps.csv.gz；含None候选的全部数据访问时间见_data_reads.csv.gz。原stdout、逐步JSONL及失败checkpoint存ignored cache，关键日志有版本化副本。

仅1完整epoch；恢复探针是影子副本下一epoch首次有效窗口的1次工程更新，不计入完整epoch预算，不覆盖模型。不能据此声称100/50轮收敛、性能复现或与作者成绩可比。仪器同步与有限性检查开销已另列。DLow的CVAE参数不在优化器中，原requires_grad/eval行为保留，状态hash用于确认未更新。
