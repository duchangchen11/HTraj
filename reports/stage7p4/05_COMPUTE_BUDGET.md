# 计算与存储预算

```json
{
  "complete_epochs_measured": true,
  "estimates_available": true,
  "timing_fit": {
    "CVAE": {
      "nonnegative_coefficients": [
        0.14583710080314433,
        0.0,
        7.540924356840097e-05
      ],
      "RMSE_seconds": 0.019494835795351943,
      "measured_people_range": [
        1.0,
        57.0
      ],
      "measured_step_sum": 506.1404369455995,
      "measured_epoch_seconds": 509.88280144898454,
      "interpretation": "Instrumented HOTEL scene-step model, not a trained prediction model; no test ADE/FDE used"
    },
    "DLow": {
      "nonnegative_coefficients": [
        0.07891857685434589,
        0.00016235771408102065,
        7.641329171800105e-05
      ],
      "RMSE_seconds": 0.013400447185603093,
      "measured_people_range": [
        1.0,
        57.0
      ],
      "measured_step_sum": 302.99125116475625,
      "measured_epoch_seconds": 307.3549300620216,
      "interpretation": "Instrumented HOTEL scene-step model, not a trained prediction model; no test ADE/FDE used"
    }
  },
  "fivefold_point_hours": 86.03893095374457,
  "fivefold_low_hours": 60.04466239427401,
  "fivefold_high_hours": 129.51579888389452,
  "HOTEL_high_hours": 28.055106060738765,
  "current_free_bytes": 878223360,
  "all_fivefold_periodic_checkpoint_bytes": 4050657133,
  "time_gate_passed": true,
  "storage_gate_passed": false,
  "storage_excludes_validation_outputs_and_failure_checkpoints": true,
  "caveats": [
    "Only HOTEL two epochs measured; no fold wall-time equality assumed",
    "Shape-dependent CUDA allocator, CPU parser, thermal/throttling/background load can alter throughput",
    "Only epoch1 observed; training convergence/stability at100/50 epochs unverified",
    "ETH source CVAE original cap32 and additional modules differ; time sensitivity is planning, not confidence interval",
    "DLow measured using epoch1 CVAE; final trained CVAE may alter numerical behavior and loss convergence",
    "Original train.py has no validation loop; proposed validation cadence is overhead scenario only"
  ]
}
```

实测仅HOTEL的CVAE/DLow各一个完整epoch，推断不能替代其它fold的真实训练。细分折数与[1,N,N²]拟合见compute_budget_by_fold.csv；文件若不存在则该估算未执行。0.7—1.5倍训练时间、0.5—2倍validation开销只是预设工程敏感区间，不是统计置信区间。

官方train.py不在epoch内评价validation，validation频率是未来计划的假设。周期checkpoint数量严格按原配置频率估计，不能删除历史或只保留有利模型来降低成本。存储预算还不含未来完整validation预测和失败现场，属于下界。RTX3080只在HOTEL出现的人数范围内实测，未进行其它fold的正式epoch。

GO时间门槛是预注册的HOTEL上界72h、五折上界336h规划筛选；实际GPU可用时间与存储准备需下一阶段用户决策。本阶段不自动启动100/50轮。


实测HOTEL纯训练线性预算：CVAE100轮=14.163h，DLow50轮=4.269h，合计18.432h；加预设validation与保存场景约18.636h（约13.005–28.055h）。五折点估计86.039h，敏感区间60.045–129.516h。

当前空闲0.818GiB；仅官方周期checkpoint估计3.772GiB，尚未包含validation预测和失败现场。本次NO-GO由存储触发，不是HOTEL训练或恢复失败。正式启动前需另行准备空间，不得删除历史成果或临时减少原保存频率。
