# 真实六通道语义输入审计

六通道实际是float32、六分类Softmax(dim=1)概率输出，并非任意logits或类别整数图。冻结分割器为Unet，head为Conv2d(16,6)→Identity→Softmax(dim=1)，eval且全部分割参数requires_grad=False。六通道值在[0,1]内，全部有限，通道和最大偏离1为3.57627869e-07；各通道有效区域均值/总体方差/范围见semantic_tensor_audit.csv。

| scene | shape | valid_height | valid_width | min_channel_value | max_channel_value | sum_mean | mean_abs_sum_minus1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bookstore_2 | [1, 6, 288, 384] | 270 | 356 | 0.000000 | 0.999997 | 1.000000 | 0.000000 |
| deathCircle_3 | [1, 6, 480, 352] | 476 | 350 | 0.000000 | 0.999995 | 1.000000 | 0.000000 |
| gates_7 | [1, 6, 512, 352] | 496 | 334 | 0.000000 | 0.999999 | 1.000000 | 0.000000 |
| hyang_7 | [1, 6, 512, 384] | 485 | 362 | 0.000000 | 0.999996 | 1.000000 | 0.000000 |
| nexus_4 | [1, 6, 448, 352] | 440 | 321 | 0.000000 | 0.999997 | 1.000000 | 0.000000 |


分割checkpoint SHA `75a728536ac766175019b9c352986f2aea55e776e5a3098ed0e4f0fdfe272194`；分割state SHA `79bd03ee82eea39723c27cba241f961669a53d2c47e7f26d4dce3b7bc949bd83`。不具备可核查的6通道类别名称/语义GT，始终只称channel0—5，不评分类正确率或断言道路/草地/障碍类别。

原始链路：cv2读取参考图像，INTER_AREA缩放0.25，右/下零边界补至32倍数，再按作者SMP resnet101 imagenet预处理。原语义输出的padding位置也含概率，并非必须为零。有效区域由实际cv2缩放shape确定，没有用GT裁定空间区域。原图SHA、预处理RGB张量SHA及语义tensor/file SHA均留档。

五个验证录像全部复用Stage2C已冻结语义，未重新执行分割器替换缓存。训练工程smoke的bookstore_0没有持久语义缓存，只在预先声明范围内计算一次分割；其两个训练目标完整预测/中间张量必须与旧Stage2C预检逐位相同，已通过。此例外不涉及验证语义重算或GT误差评分。

证据范围：HTraj-YNet-SDD-Development-v1，998个既有开发验证目标窗口、5个既有录像、8/12帧、0.4秒，误差单位SDD原参考像素px。数据不是ETH/UCY、不与米制论文指标混排；只有5个录像，998窗口不能视为998个独立场景。这里只报告配对描述统计与源录像敏感性，不报告窗口独立性显著性。

冻结Epoch30 checkpoint SHA `0f300417f1fb84a1f6e4d2504bdf37fe4b467a4295be22a644a74d06b228d312`，原S0预测SHA `2ddf31b50edcb8ac1a72f4c39e0b56f971d8971490559e1c829b0950ed5bc6ab`，作者MIT源码 `a345ada1557f53dc2acb4c07f71d340d366980da`。总参数53155339，预测网络1641381；全部state_dict严格加载并在进程结束验证不变。0训练、0参数更新、0测试集加载或评价；历史保护仅做文件SHA流式核验，不解析测试目标或查询测试误差。GT只在全部六组新预测已保存并核验后用于CPU评分/离线绘图。

原始表格：[results/tables/stage11c0](../../results/tables/stage11c0/)。全仓pytest：671 PASS、1 SKIP、0 FAIL、0 ERROR；完整记录full_tests.txt/XML，SKIP不计PASS。 92464项历史文件SHA全部未改变。
