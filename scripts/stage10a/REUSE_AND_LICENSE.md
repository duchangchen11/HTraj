# Stage10A 来源与接口边界

本阶段图、风险函数、GRU框架及测试为独立实现；未复制或改写第三方冻结模型。`src/risk_dynamic_graph/`是用户指定的独立Stage10命名空间，脚本和配置分别位于`stage10a/`。

只读调用HTraj自有`src/data/preprocessing.py::Recording`、`src/socialvae_causal/data.py::history_context/CausalDataset/array_sha`。输入接口为原始四列frame/person/x/y、0.4秒网格；只读历史函数仅查询窗口内8帧，完整未来只用于生成监督目标与核对GT SHA。其源码SHA进入交付清单。不调用原TOHI/SocialVAE/Re模型，不复用旧训练缓存或模型权重。

Snapshot仅供方法审查：官方仓库`704e93339c58e15740c786881add06de8a397ba3`，Apache-2.0。以`--no-checkout`克隆后通过`git show`读取指定blob，未检出权重，未运行作者模型。仓库工作树显示D是未checkout的预期状态，不是删除作者文件或修改旧仓库。源码证据留在忽略cache，不再分发。

Social-STGCNN及SGCN旧冻结源码为MIT，Trajectron++官方源码为MIT（本阶段未导入）；HighGraph/SocialCircle仅审查论文。不推断论文开放下载等于允许重新分发全文。六篇PDF仅在`experiments/stage10a/cache/papers/`，Git提交链接、hash和自行撰写的分析。

Python3.10运行环境只读复用现有Stage8隔离venv及Stage4依赖；未安装或修改环境。PyTorch/NumPy等运行依赖保留各自许可证。HTraj根目录未声明统一开源许可证，本阶段不擅自为历史成果重新授权。
