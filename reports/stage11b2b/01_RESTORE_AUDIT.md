# Epoch30→Epoch50 连续训练恢复审计

本阶段仅作 CONVERGENCE_DIAGNOSTIC_ONLY；所有误差单位为 SDD 原始参考图像像素。数据、模型、Adam、学习率、损失、训练顺序、精度与采样实现保持 Stage11B-1C 数值协议。

起点为 `experiments/stage11b1c/cache/checkpoints/epoch_030.pt`，预期 SHA256 `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`。CPU weights_only 安全加载及 strict 模型键、90组 Adam、每组56460步、数据及配置SHA检查见 `checkpoint_static_audit.json`。原始源码commit为 `a345ada1557f53dc2acb4c07f71d340d366980da`。

旧 `scripts/stage11b0/training_adapter.py::training_loss` 在每次计算损失时调用 `set_training`；该函数调用 net.train() 并将冻结分割器设为 eval。旧验证阶段的 net.eval() 不会使后续 batch 保持 eval。预测核心为Conv/ReLU/Pool/插值/SoftArgmax，GPU准入还会实际检查模式依赖模块和输出。若存在实质数值模式错误则停止，不在延长中修复。

GPU准入必须经原 `load_checkpoint` 同时恢复模型、Adam与Python/NumPy/CPU/CUDA RNG；在临时副本验证保存恢复后的下一次Adam更新逐位相同、K20反序/缓存重算及未来GT反事实。正式进程再次从原Epoch30恢复，绝不继承临时更新。完成记录见 `results/tables/stage11b2b/resume_admission.json`（执行后生成）。

训练上限Epoch50，新增20轮/37640步，累计94100步。仅在35/40/45/50各全量998目标K20验证，原7次完整验证参与相同minADE20选模。H1阈值严格按 `configs/stage11b2b/analysis.json` 固定；新增GPU进程累计硬限16200秒、安全阈值15000秒，含启动、加载、检查、验证、保存与退出。

前置CPU审计补加优化器摘要时，两次独占写入触发 FileExistsError，未执行GPU或训练；失败日志原样保留 `prepare_recheck_stdout.txt`、`prepare_final_stdout.txt`。随后增加“已存在则校验完全相同”行为并重新通过，未修改旧文件或数值计算。

第一次GPU准入恢复状态成功，随后额外teacher-forcing检查的合成GT被平移100000px，越过热图模板范围，触发旧代码Unrepresentable patch断言。未进行正式训练或Adam更新。失败进程耗时7.990秒纳入预算，原日志与admission记录保留failed01命名。仅将该训练用合成扰动改回旧准入的末点10px；历史预测接口的百万像素未来GT反事实仍保持。重新冻结代码和配置后复核，未改模型、loss、数据或正式训练数值。
