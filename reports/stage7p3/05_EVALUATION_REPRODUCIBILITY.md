# 评价一致性与执行可追溯性

正式推理冻结提交：`f672731ecd625c39583100fc69be43ae0020ec99`。该提交实际成功后，入口核对git工作区干净、协议/配置/执行源码已真实进入提交，才写execution_start并开始推理。

原test.py的get_model_prediction、save_prediction、test_model函数体通过AST原样执行；不用原__main__的strict=False和自动评分。DLow构造器严格加载指定CVAE，wrapper严格加载指定DLow，并逐张量验证CVAE一致。未修改原网络、loss、数据划分或数据字节。

全体3348个窗口先通过global_output_pre_metric_validation.json的ID/frame/K完整性检查，期间不计算任何ADE/FDE，再启动统一评分。官方eval.py通过原始CLI执行。独立核验另建(frame, person_id)字典对齐GT，不调用官方align_gt；要求每个候选均含同一完整12帧×目标集合，恰好20个文件，且所有坐标有限。两者读取同一samples文件；以原生float32官方函数重算全精度值，独立float64核验。

五折官方全精度与独立指标最大差：1.3060846848489405e-09米；冻结容差1e-5米。CLI四位小数容差0.000051米。三位小数预测序列化是官方流程本身，两套评分均在序列化后进行，不将其误称为网络未量化的原始输出。

环境：Python 3.10.21，PyTorch 2.5.1+cu124，CUDA 12.4，NumPy 1.26.4，cuDNN 90100；NVIDIA GeForce RTX 3080, 10240 MiB, 570.133.07。cudnn benchmark=False、deterministic=False。未声称旧版本PyTorch与现环境逐位等同。

兼容改动只在运行时提供_LinearWithBias与np.int旧名称；官方源码保持冻结。种子按每折原配置seed=1；DLow mean=True的头并非随机CVAE采样。

PyTorch allocated/reserved为每折精确峰值；整卡显存为名义等待0.1秒的nvidia-smi轮询峰值（另有命令耗时），可能漏掉更短峰值。计时包含原始test_model两次模型调用及三类文件保存；model_only为两次模型调用、cuda同步的总和，不假装单次纯forward吞吐。

失败记录：首次预检JSON写入因np.int64失败，已保留日志，修复仅计数序列化；首次git检查被官方YAML末尾空行阻止，推理未启动，局部attributes保留配置SHA后成功提交。推理与评分的实际异常见inference_completion与evaluation_summary，未发生项目不虚构异常。

完整测试：419项，failures=0，errors=0，skipped=0。历史SHA核验11348项，旧文件变化0项。完整JUnit和控制台记录均归档。

全部样本文件（包括不参与评分的recon和GT）路径/SHA/字节数见各折prediction_file_SHA256.csv.gz；官方数据74份、源码43份、10权重及10配置SHA见冻结协议。预测大文件只存ignored cache，不提交Git。

来源：固定官方版本的`test.py`、`eval.py`、`model/dlow.py`及配置；本地执行证据`results/tables/stage7p3/`。公开参照见[作者维护版README](https://github.com/Khrylx/AgentFormer/tree/e4fe8dd6df3b3d2033665c5f8ac3544e81c44db5)。

验收工具补记：逐文件启动git check-ignore的末尾检查因效率过低被SIGINT停止，原脚本与中断日志保留在verify_results.py及final_artifact_checks_execution.txt。相同文件SHA与忽略规则检查改由verify_results_batched.py批量执行；这是验收工具修订，未重跑模型、未修改预测或指标。实际完成状态见final_artifact_checks.json及artifact_verifier_interruption.json。最终交付源码与报告SHA见final_source_versions.json；post_execution_integrity.json记录的是当次核验时的版本。
