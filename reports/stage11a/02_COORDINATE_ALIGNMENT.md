# 原始标注、像素坐标和参考图对齐

结果为**PARTIAL几何核验，不是原视频精确配准或语义对齐已经证明**。六个真实录像样例均有对应参考图与原格式标注；镜像来源仍UNVERIFIED。完整过程见`scripts/stage11a/sdd_tools.py`和alignment_audit.py，逐文件SHA/矩阵/统计见alignment_results.json；无虚构轨迹。

## 字段与坐标定义

每行10列：track_id，xmin，ymin，xmax，ymax，frame，lost，occluded，generated，带引号label。track_id只在该录像内标识轨迹；同人跨录像身份不推断。lost=1表示出视野；occluded=1表示遮挡；generated=1是标注系统插值，而非本轮神经网络伪标签。坐标为参考图左上原点、x向右/y向下的像素，中心为((xmin+xmax)/2,(ymin+ymax)/2)。

新审计parser明确header=None，保留首行；作者Y-Net load_SDD用header=0且names，会把无头原文件首行丢掉，这是源码可定位差异。本轮不改作者loader，也不声称新parser就是官方精确预处理复现。

## 真实覆盖

|scene/video|原图宽×高|已读标注行|lost=0中心越界/总数|历史track/frame|缩放往返最大误差px|
|---|---|---|---|---|---|
|bookstore/video0|1424×1088|100059|0/66620|2, 6881–6965|1.14e-13|
|coupa/video0|1980×1093|98637|0/75272|1, 4030–4114|2.84e-14|
|deathCircle/video0|1630×1948|102712|0/53889|1, 9250–9334|2.27e-13|
|gates/video0|1325×1973|102527|0/56578|8, 311–395|1.14e-13|
|hyang/video0|1455×1925|100651|0/70072|0, 11116–11200|1.14e-13|
|little/video0|1417×2019|51462|0/24517|0, 391–475|2.27e-13|

各图位于`results/tables/stage11a/alignment_figures/`，包含原图、缩放图和原像素局部crop，标注scene/video/frame/track、原图尺寸、坐标单位及变换。初版短位移轨迹被帧标号覆盖、可读性不足，因此保留于alignment_figures_initial；终版选择首条连续8点且净移动至少20px的真实历史，仅为图形可读性，不是预测分组、数据筛选或改善案例。青色起点/黄色终点，所有点仍是原标注行，未追加插值。

6/6样例的lost=0中心均在参考图边界内；near-boundary数量另存，原lost行和occluded/generated数量没有被隐瞒。这是对已下载前缀的统计，不能外推全部60录像。部分图像在道路/步道附近，另有树冠/绿化边缘歧义；reference的采样帧未知，没有对应原视频帧，不能证明每个框时序位置准确、亚像素标定或所有轨迹都处于可通行区。

本轮假设annotation→reference为恒等像素映射，未引入未核实homography。缩放使用每幅图实际输出宽高：T=diag(W'/W,H'/H,1)，逆变换往返误差小于1e-9px；这只是数学自洽，**不是配准误差小于1e-9px**。Y-Net把轨迹乘统一resize，再对图像OpenCV缩放、右/下padding32；尺寸四舍五入及像素中心约定带来亚像素差，独立结果在resize_convention_audit.csv，不能把它与本轮PIL边缘坐标矩阵无声等同。

六个录像分别使用各自尺寸/矩阵，未共享世界变换。人工逐图检查发现，deathCircle/video0的track1历史投影接近树冠/绿化边缘，hyang/video0也有阴影/遮挡邻近情况。没有语义GT及原视频时，这只能记录为场景解释歧义，不能判定轨迹错误、真实不可通行或模型失效；它也说明不能把tree/terrain标签直接当确定障碍。

没有米制标定，不使用近似estimated_scales来宣称精确米定位。语义图未取得，D项语义对齐无法执行；不把RGB静态场景当作D通过。
