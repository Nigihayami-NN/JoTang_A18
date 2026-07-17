# RGBT 目标检测系统

本项目是一个 **RGBT (可见光 + 红外热成像)** 目标检测系统。项目基于 Ultralytics YOLO 架构进行深度定制，支持 4 通道输入（RGB 3通道 + IR 1通道）的旋转目标检测（OBB），并包含了一套与 Java 后端高效通信的分布式推理引擎。

## ✨ 核心特性

- **四通道 RGBT 融合 (4-Channel Fusion)**：支持将常规可见光图像 (RGB) 与红外热成像图像 (IR) 在通道维度级联拼接，大幅提升在夜间、雾霾等复杂场景下的目标检测鲁棒性。
- **自定义 OBB 旋转框训练**：提供专门针对无人机/航拍视角优化的训练脚本 (`m_train.py`)，支持混合精度训练、多尺度训练。
- **模型权重无缝移植**：内置官方 3 通道模型预训练权重到 4 通道自定义模型的自动化映射与移植逻辑，解决冷启动训练收敛慢的问题。
- **高性能推理 Worker**：提供 `cnct.py` 和 `backend.py` 作为分布式推理节点，通过轮询 Java 任务分发服务端，自动拉取双流图片并在本地进行 FP16 半精度推理，结果实时回传。

## 📁 目录结构与数据集规范

模型训练依赖于特定的数据集组织格式。请在 `./CustomDataSet.yaml` 中配置好类别编号和路径。

数据集存放于 `./CustomDataSet/` 目录下，需严格按照以下结构组织：

```text
JoTang_A18
├── ultralytics/        # YOLO 核心库
├── CustomDataSet/      # 你的数据集目录
│   ├── Images/
│   │   ├── train/      # 训练集图片
│   │   ├── val/        # 验证集图片
│   │   └── test/       # 测试集图片
│   └── Labels/
│       ├── train/      # 训练集标签 (YOLO OBB 格式)
│       ├── val/        # 验证集标签
│       └── test/       # 测试集标签
├── m_train.py          # 核心训练脚本 (4通道 OBB)
├── cnct.py             # 生产环境推理 Worker (5分类)
├── backend.py          # 开发测试环境推理 Worker
└── requirements.txt    # 环境依赖
```

## 🚀 快速开始

### 1. 环境准备

推荐使用 Python 3.8+ 及 PyTorch 2.0+ 环境。

```bash
pip install -r requirements.txt
```

### 2. 模型训练

确认已经按照上述结构准备好数据集，并修改了 `CustomDataSet.yaml` 配置文件。随后直接运行：

```bash
python m_train.py
```
> **Note**: `m_train.py` 内含针对大显存（如 RTX 4090）的显存优化（如 `gradient_checkpointing`）与超参数调优，如果你使用显存较小的显卡，请按需调小 `batch` 尺寸并设置 `accumulate`。

### 3. 启动推理 Worker

在 `cnct.py` 中配置好 `JAVA_SERVER_URL`（后端服务器地址）与 `WEIGHT_PATH`（训练好的模型权重路径）。当前推理脚本默认支持 5 类别检测（`car`, `truck`, `bus`, `van`, `freight_car`）。

```bash
python cnct.py
```

推理引擎启动后将：
1. 自动进行模型预热（FP16）。
2. 高频轮询拉取后端的推理任务。
3. 并发下载 RGB 和 IR 图像并完成四通道拼接预处理。
4. 推理并将结果组装为 JSON 格式通过 HTTP POST 提交回 Java 后端。

## 💡 开发说明

- **图像通道说明**：在推理时，服务端传入的红外图会被强制转换为单通道灰度图，随后与 3 通道 RGB 图像进行 `np.concatenate` 合并成 4 通道输入，请确保模型本身为 4 通道输入架构。
- **内存回收**：训练代码中通过 `on_train_epoch_end` 回调函数强制在每一个 epoch 结束时执行 `gc.collect()` 与 `torch.cuda.empty_cache()`，有效防止由于长期训练导致的 OOM 问题。

