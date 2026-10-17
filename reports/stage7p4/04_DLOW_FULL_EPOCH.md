# DLow完整epoch成本测量

实际执行状态：success=True。

```json
{
  "P50_seconds": 0.08438408249639906,
  "P90_seconds": 0.15101196581963452,
  "P99_seconds": 0.23705656939826436,
  "RNG_saved": true,
  "all_step_mean_seconds": 0.09717487208619507,
  "author_initialization": false,
  "candidate_frame_slots": 4349,
  "checkpoint": "/home/lrj/HTraj/experiments/stage7p4/cache/execution_v2/training_workspace/results/hotel_agentformer/models/model_0001.p",
  "checkpoint_SHA256": "fa5512f051b92eca1a0b3779a96fcd2ad4b4d5f5666b10b2e8bc3052b5ac4c1d",
  "checkpoint_bytes": 7139056,
  "checkpoint_save_seconds": 0.024702645023353398,
  "context_truncated": false,
  "external_CVAE_state_before": {
    "parameter_L2": 302.0719377742081,
    "parameter_count": 5972098,
    "parameter_max_abs": 1.0378286838531494,
    "parameter_state_sha256": "9287cee08accd91b9a25ebb02ad631e785610c8d053832ff65079557939871e3",
    "parameters_finite": true
  },
  "external_CVAE_unchanged": true,
  "final_state": {
    "parameter_L2": 21.700889911062564,
    "parameter_count": 591872,
    "parameter_max_abs": 0.12783896923065186,
    "parameter_state_sha256": "65491a064629d9fc11afac192234b195bf4ce22c2778067ed9c5d2d6d18fd0f7",
    "parameters_finite": true
  },
  "full_epochs_completed": 1,
  "full_target_windows": 29676,
  "initial_state": {
    "parameter_L2": 21.324592835032917,
    "parameter_count": 591872,
    "parameter_max_abs": 0.062499888241291046,
    "parameter_state_sha256": "d839a51485af7dfbd7ead0acef98cb1bb4e4be89ba6e1e5969378d5af04a3b1d",
    "parameters_finite": true
  },
  "learning_rate": 0.0001,
  "loader_initialization_seconds": 0.7303715539746918,
  "max_context_people": 57,
  "official_configured_epochs": 50,
  "official_training_source_SHA256": "5774ccaca4043b67e80fa35a5651be7397a6a0fc1de4e2d61dc6c92cdfff4853",
  "optimizer_updates": 3118,
  "original_config": "hotel_agentformer",
  "random_initialization": true,
  "raw_log": "/home/lrj/HTraj/experiments/stage7p4/cache/execution_v2/DLow/official_train_stdout.txt",
  "raw_log_SHA256": "2138cfea04dca7bab7e9bb974cdc32153f939c66fdc8baf1012444029f634827",
  "resource": {
    "OOM": false,
    "peak_allocated_bytes": 3982871040,
    "peak_reserved_bytes": 9642704896,
    "seconds": 307.3549300620216,
    "whole_card_memory_samples": 2260,
    "whole_card_sampled_peak_MiB": 8982,
    "whole_card_sampling_interval_seconds": 0.1
  },
  "resume_probe": {
    "RNG_exact": true,
    "base_CVAE_unchanged": true,
    "finite_loss": 18.358043670654297,
    "frame": 128,
    "gradient_norm": 142.05072021484375,
    "model_state_after": {
      "parameter_L2": 21.700752241198476,
      "parameter_count": 591872,
      "parameter_max_abs": 0.1278323382139206,
      "parameter_state_sha256": "933fe7c3b7f58ddc386fe25cf8076ae6e80bf571cd745b49091815503a339cd6",
      "parameters_finite": true
    },
    "model_state_before": {
      "parameter_L2": 21.700889911062564,
      "parameter_count": 591872,
      "parameter_max_abs": 0.12783896923065186,
      "parameter_state_sha256": "65491a064629d9fc11afac192234b195bf4ce22c2778067ed9c5d2d6d18fd0f7",
      "parameters_finite": true
    },
    "not_an_extra_epoch": true,
    "optimizer_entries": 8,
    "optimizer_state_exact": true,
    "optimizer_step_max_after": 3119.0,
    "optimizer_step_max_before": 3118.0,
    "optimizer_step_min_after": 3119.0,
    "optimizer_step_min_before": 3118.0,
    "optimizer_update_probe": 1,
    "people": 4,
    "scheduler_state_exact": true,
    "seconds": 1.3793227479909547,
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
    "step_size": 5,
    "verbose": false
  },
  "seed": 1,
  "stage": "DLow",
  "stage_execution_epochs": 1,
  "steady_mean_seconds": 0.0970844876518309,
  "steady_step_count": 3108,
  "steady_warmup_updates_excluded": 10,
  "success": true,
  "tensorboard_scalar_calls": 828,
  "total_data_read_seconds": 39.922583743347786,
  "total_gradient_and_parameter_check_seconds": 12.274228655092884,
  "updates": 3118,
  "valid_windows": 3118
}
```

每次有效窗口的原始loss分项、梯度范数、上下文人数、同步step时间及显存见同名_steps.csv.gz；含None候选的全部数据访问时间见_data_reads.csv.gz。原stdout、逐步JSONL及失败checkpoint存ignored cache，关键日志有版本化副本。

仅1完整epoch；恢复探针是影子副本下一epoch首次有效窗口的1次工程更新，不计入完整epoch预算，不覆盖模型。不能据此声称100/50轮收敛、性能复现或与作者成绩可比。仪器同步与有限性检查开销已另列。DLow的CVAE参数不在优化器中，原requires_grad/eval行为保留，状态hash用于确认未更新。
