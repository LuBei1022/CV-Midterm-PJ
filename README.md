# CV-Midterm-PJ

本仓库为计算机视觉课程期中项目代码仓库，包含三个任务的代码实现，分别对应图像分类、目标检测与多目标追踪、图像分割三个方向。

项目主要使用 Python 和 PyTorch 实现。其中，任务一和任务三基于 Oxford-IIIT Pet Dataset 完成宠物图像分类与宠物图像分割，任务二使用 YOLOv8 完成无人机视角下的目标检测，并结合 ByteTrack 完成多目标追踪与车辆越线计数。

---

## 一、项目结构

```text
CV-Midterm-PJ/
├── task1/
│   ├── DataLoader.py
│   ├── model.py
│   ├── train.py
│   ├── tune.py
│   ├── ablation_study.py
│   ├── attention_study.py
│   ├── ablation_study_results.png
│   ├── baseline_training_curve.png
│   ├── hyperparameter_comparison.png
│   └── se_attention_comparison.png
│
├── task2/
│   ├── train_model.py
│   ├── tracker.py
│   ├── requirements.txt
│   └── report_figures/
│
├── task3/
│   ├── DataLoader.py
│   ├── model.py
│   ├── losses.py
│   ├── train.py
│   └── experiments.py
│
├── README.md
└── LICENSE
```

---

## 二、整体环境说明

本项目主要使用以下环境：

```text
Python >= 3.9
PyTorch
torchvision
numpy
opencv-python
Pillow
matplotlib
scikit-learn
ultralytics
swanlab
```

任务二的依赖可以直接通过 `requirements.txt` 安装：

```bash
pip install -r task2/requirements.txt
```

任务一和任务三如果运行时报缺少依赖，可以根据报错手动安装，例如：

```bash
pip install torch torchvision numpy pillow matplotlib scikit-learn swanlab
```

---

# Task 1：图像分类任务

## 1.1 任务简介

任务一主要完成宠物图像分类任务。该任务基于 PyTorch 搭建图像分类模型，在 Oxford-IIIT Pet Dataset 数据集上进行训练和测试。

本任务使用在 ImageNet 上预训练的 ResNet-18 作为 Baseline，修改最后的输出层以适配宠物类别数，并进行微调训练。同时，本任务还包含超参数调优、预训练消融实验和注意力机制对比实验。

## 1.2 主要文件说明

| 文件 | 说明 |
|---|---|
| `task1/DataLoader.py` | 数据读取与预处理 |
| `task1/model.py` | 分类模型结构定义 |
| `task1/train.py` | 基础模型训练代码 |
| `task1/tune.py` | 超参数调优实验代码 |
| `task1/ablation_study.py` | 消融实验代码 |
| `task1/attention_study.py` | 注意力机制对比实验代码 |
| `task1/*.png` | 实验结果图与训练曲线 |

## 1.3 数据集准备

任务一使用 Oxford-IIIT Pet Dataset 作为图像分类数据集。由于完整数据集文件较大，本仓库不直接上传数据集。运行代码前，需要自行下载数据集并放置到对应目录。

Oxford-IIIT Pet Dataset 官方下载地址：

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/
```

官方页面中需要下载以下两个文件：

```text
images.tar.gz
annotations.tar.gz
```

其中：

- `images.tar.gz`：包含所有宠物图像；
- `annotations.tar.gz`：包含类别标签、训练测试划分文件、XML 标注文件以及 trimap 分割标注。

下载完成后，将两个压缩包解压。解压后通常会得到以下结构：

```text
images/
annotations/
├── list.txt
├── trainval.txt
├── test.txt
├── trimaps/
└── xmls/
```

任务一主要使用：

```text
images/
annotations/list.txt
annotations/trainval.txt
annotations/test.txt
```

其中：

- `images/` 保存所有宠物图片；
- `list.txt` 保存图像名称、类别编号、物种信息等标注；
- `trainval.txt` 和 `test.txt` 可用于训练集、验证集和测试集划分；
- Oxford-IIIT Pet Dataset 共包含 37 个宠物品种类别，因此分类模型最终输出类别数应设置为 37。

建议将数据集放置为以下结构：

```text
task1/
└── data/
    └── oxford-iiit-pet/
        ├── images/
        └── annotations/
            ├── list.txt
            ├── trainval.txt
            ├── test.txt
            ├── trimaps/
            └── xmls/
```

如果本地数据集路径与上述结构不同，需要在 `task1/DataLoader.py` 或对应训练脚本中修改数据集路径。

也可以使用 `torchvision.datasets.OxfordIIITPet` 自动下载数据集，例如：

```python
from torchvision.datasets import OxfordIIITPet

train_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="category",
    download=True
)

test_dataset = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="category",
    download=True
)
```

使用该方式时，`target_types="category"` 表示读取 37 类宠物分类标签。

## 1.4 运行方法

进入任务一目录：

```bash
cd task1
```

运行基础训练代码：

```bash
python train.py
```

运行超参数调优实验：

```bash
python tune.py
```

运行消融实验：

```bash
python ablation_study.py
```

运行注意力机制对比实验：

```bash
python attention_study.py
```

---

# Task 2：目标检测、多目标追踪与车辆计数任务

## 2.1 任务简介

任务二主要完成无人机视频序列中的车辆检测、追踪与越线计数。

本任务使用 YOLOv8 作为目标检测模型，并使用 ByteTrack 进行多目标追踪。程序会读取连续视频帧，对车辆目标进行检测和 ID 分配，并通过设置计数线统计穿越计数线的目标数量。

任务二主要实现内容包括：

1. 使用 YOLOv8 对 VisDrone 数据集进行微调训练；
2. 使用训练后的 `best.pt` 权重进行目标检测；
3. 使用 ByteTrack 对连续帧中的目标进行 ID 跟踪；
4. 在视频画面中绘制检测框、类别、置信度和目标 ID；
5. 设置计数线并统计穿越计数线的车辆数量；
6. 输出最终带有检测、追踪和计数结果的视频。

## 2.2 主要文件说明

| 文件 | 说明 |
|---|---|
| `task2/train_model.py` | YOLOv8 模型训练与微调代码 |
| `task2/tracker.py` | 目标检测、多目标追踪与车辆计数代码 |
| `task2/requirements.txt` | 任务二所需 Python 依赖 |
| `task2/report_figures/` | 实验报告中使用的结果截图 |

---

## 2.3 任务二环境配置

建议单独为任务二创建 Python 环境，避免不同任务之间的依赖冲突。

### 方法一：使用 Conda 创建环境

```bash
conda create -n cv-task2 python=3.9
conda activate cv-task2
```

### 方法二：使用 venv 创建环境

Windows：

```bash
python -m venv cv-task2-env
cv-task2-env\Scripts\activate
```

macOS / Linux：

```bash
python -m venv cv-task2-env
source cv-task2-env/bin/activate
```

### 安装任务二依赖

进入任务二目录：

```bash
cd task2
```

安装依赖：

```bash
pip install -r requirements.txt
```

`requirements.txt` 中主要包含以下依赖：

```text
ultralytics
opencv-python
numpy
torch
torchvision
```

如果安装 PyTorch 时遇到 CUDA 版本不匹配的问题，可以根据自己电脑的 CUDA 版本到 PyTorch 官网选择对应安装命令。

---

## 2.4 数据集准备

任务二使用 VisDrone 数据集。由于完整数据集和训练权重文件较大，因此本仓库不直接上传数据集和模型权重。运行代码前，需要自行下载数据集并放置到对应目录。

本任务主要涉及两部分数据：

1. **YOLOv8 训练数据**：使用 VisDrone-DET 数据集进行目标检测模型微调；
2. **追踪与计数测试数据**：使用 VisDrone-MOT 数据集中的测试视频帧序列进行目标追踪和车辆计数。

### 2.4.1 数据集下载地址

VisDrone 官方数据集地址：

```text
https://github.com/VisDrone/VisDrone-Dataset
```

在该页面中可以下载不同任务对应的数据集。

---

### 2.4.2 YOLOv8 训练数据

如果需要重新训练 YOLOv8 检测模型，需要下载 VisDrone-DET 数据集，至少包括：

```text
VisDrone2019-DET-train
VisDrone2019-DET-val
```

也可以下载：

```text
VisDrone2019-DET-test-dev
```

用于后续测试。

YOLOv8 训练需要使用 YOLO 格式的数据标注，因此原始 VisDrone 标注需要转换为 YOLO 格式。可以使用 Ultralytics 提供的 `VisDrone.yaml` 自动完成数据下载和格式转换，也可以自行将 VisDrone 原始标注转换为 YOLO 格式。

Ultralytics VisDrone 数据集说明：

```text
https://docs.ultralytics.com/zh/datasets/detect/visdrone/
```

训练时可在 `train_model.py` 中指定数据配置文件，例如：

```python
model.train(
    data="VisDrone.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    workers=4,
    project="CV_Midterm_Runs",
    name="finetune_20_epochs"
)
```

训练完成后，最优模型权重 `best.pt` 会保存在训练输出目录中。常见路径示例：

```text
CV_Midterm_Runs/finetune_20_epochs/weights/best.pt
```

或：

```text
runs/detect/CV_Midterm_Runs/finetune_20_epochs/weights/best.pt
```

实际路径会受到 `train_model.py` 中 `project` 和 `name` 参数的影响。因此，运行追踪代码前，需要以自己电脑上实际生成的 `best.pt` 路径为准，并修改 `tracker.py` 中的 `model_path`。

---

### 2.4.3 追踪与计数测试数据

目标追踪与车辆计数部分使用 VisDrone-MOT 数据集中的测试序列。需要下载：

```text
VisDrone2019-MOT-test-dev
```

解压后目录结构示例：

```text
VisDrone2019-MOT-test-dev/
└── VisDrone2019-MOT-test-dev/
    └── sequences/
        └── uav0000355_00001_v/
            ├── 0000001.jpg
            ├── 0000002.jpg
            ├── 0000003.jpg
            └── ...
```

本项目的 `tracker.py` 默认读取其中一个测试序列：

```python
input_folder = r"VisDrone2019-MOT-test-dev\VisDrone2019-MOT-test-dev\sequences\uav0000355_00001_v"
```

如果数据集存放位置不同，需要将 `input_folder` 修改为自己电脑上的实际路径。

---

### 2.4.4 本地路径修改说明

运行 `tracker.py` 前，需要重点检查以下两个路径：

```python
model_path = r"runs\detect\CV_Midterm_Runs\finetune_20_epochs\weights\best.pt"
input_folder = r"VisDrone2019-MOT-test-dev\VisDrone2019-MOT-test-dev\sequences\uav0000355_00001_v"
```

其中：

- `model_path` 表示训练完成后生成的 YOLOv8 最优权重路径；
- `input_folder` 表示用于追踪和计数的视频帧序列路径。

如果运行时出现模型无法读取或图片帧无法读取的问题，优先检查这两个路径是否正确。

---

## 2.5 YOLOv8 模型训练

进入任务二目录：

```bash
cd task2
```

运行训练脚本：

```bash
python train_model.py
```

训练脚本会对 YOLOv8 模型进行微调。训练完成后，需要在输出文件夹中找到最优权重文件：

```text
best.pt
```

然后将 `tracker.py` 中的 `model_path` 修改为该文件的实际路径。

---

## 2.6 目标追踪与车辆计数

完成模型训练并确认 `best.pt` 路径正确后，运行追踪代码：

```bash
cd task2
python tracker.py
```

程序会完成以下流程：

1. 读取训练后的 YOLOv8 权重；
2. 读取 VisDrone 测试序列中的连续图像帧；
3. 对每一帧进行目标检测；
4. 使用 ByteTrack 对检测到的目标进行 ID 分配；
5. 根据目标中心点轨迹判断是否穿越计数线；
6. 在输出视频中绘制检测框、类别、ID 和计数结果；
7. 保存最终结果视频。

输出视频示例：

```text
output_final_counted1.mp4
```

---

## 2.7 关于 ID 生成与追踪原理说明

任务二中的目标 ID 不是手动设置的，也不是简单随机生成的，而是由 ByteTrack 追踪器根据连续帧中的检测结果自动分配和维护。

ByteTrack 会根据目标检测框的位置、置信度、运动连续性以及相邻帧之间的匹配关系，将同一个目标在不同帧中的检测结果关联起来。如果目标在连续帧中能够被稳定检测到，并且位置变化较为连续，则通常可以保持相同的 ID。

但是在以下情况下，可能会出现目标丢失或 ID 跳变：

- 目标被其他车辆、建筑物或画面边缘遮挡；
- 目标尺寸较小，检测置信度较低；
- 画面中车辆密集，目标之间距离较近；
- 无人机视角变化导致目标外观变化明显；
- 检测模型在某些帧中未能检测到该目标；
- 相邻目标的检测框发生重叠，导致追踪器匹配错误。

因此，追踪结果的稳定性与检测模型效果、视频画面质量、目标密集程度、遮挡情况以及计数线位置都有关系。

---

# Task 3：图像分割任务

## 3.1 任务简介

任务三主要完成宠物图像分割任务。该任务基于 U-Net 结构实现语义分割模型，对图像中的宠物主体、背景和边界区域进行像素级分类。

本任务不使用任何预训练权重，而是从随机初始化开始训练 U-Net。为了保证在无预训练条件下的收敛速度，任务三复用 Oxford-IIIT Pet Dataset 中的三分类 trimap 分割标注。

## 3.2 主要文件说明

| 文件 | 说明 |
|---|---|
| `task3/DataLoader.py` | 分割数据集读取与预处理 |
| `task3/model.py` | U-Net 分割模型结构定义 |
| `task3/losses.py` | Dice Loss 等损失函数定义 |
| `task3/train.py` | 分割模型训练代码 |
| `task3/experiments.py` | 不同实验设置与结果对比代码 |

## 3.3 数据集准备

任务三同样使用 Oxford-IIIT Pet Dataset，但与任务一不同，任务三使用的是其中的像素级 trimap 分割标注。

Oxford-IIIT Pet Dataset 官方下载地址：

```text
https://www.robots.ox.ac.uk/~vgg/data/pets/
```

官方页面中需要下载以下两个文件：

```text
images.tar.gz
annotations.tar.gz
```

其中：

- `images.tar.gz`：包含所有宠物图像；
- `annotations.tar.gz`：包含分类标注、训练测试划分文件、XML 标注文件以及 `trimaps/` 分割标注。

任务三主要使用：

```text
images/
annotations/trimaps/
annotations/trainval.txt
annotations/test.txt
```

其中：

- `images/` 保存输入图像；
- `annotations/trimaps/` 保存每张图像对应的三分类分割 mask；
- `trainval.txt` 和 `test.txt` 可用于训练集、验证集和测试集划分。

建议将数据集放置为以下结构：

```text
task3/
└── data/
    └── oxford-iiit-pet/
        ├── images/
        └── annotations/
            ├── list.txt
            ├── trainval.txt
            ├── test.txt
            ├── trimaps/
            └── xmls/
```

Oxford-IIIT Pet Dataset 的 trimap 标注通常包含三类像素：

```text
1: 宠物主体
2: 背景
3: 边界
```

在模型训练时，可以将原始标签减 1，转换为更适合 PyTorch 训练的类别编号：

```text
0: 宠物主体
1: 背景
2: 边界
```

如果本地数据集路径与上述结构不同，需要在 `task3/DataLoader.py` 或训练脚本中修改数据集路径。

也可以使用 `torchvision.datasets.OxfordIIITPet` 自动下载数据集，例如：

```python
from torchvision.datasets import OxfordIIITPet

train_dataset = OxfordIIITPet(
    root="./data",
    split="trainval",
    target_types="segmentation",
    download=True
)

test_dataset = OxfordIIITPet(
    root="./data",
    split="test",
    target_types="segmentation",
    download=True
)
```

使用该方式时，`target_types="segmentation"` 表示读取分割任务所需的 trimap mask。

## 3.4 模型说明

任务三使用 U-Net 作为主要分割模型。U-Net 包含编码器、解码器和跳跃连接结构，可以较好地融合浅层空间信息和深层语义信息。

模型输出为三类分割结果：

```text
0: 宠物主体
1: 背景
2: 边界
```

## 3.5 损失函数

任务三支持不同损失函数设置，例如：

```text
ce       : Cross Entropy Loss
dice     : Dice Loss
combined : Cross Entropy Loss + Dice Loss
```

可以在 `task3/train.py` 中修改损失函数类型：

```python
LOSS_TYPE = "combined"
```

## 3.6 运行方法

进入任务三目录：

```bash
cd task3
```

运行训练代码：

```bash
python train.py
```

运行实验对比代码：

```bash
python experiments.py
```

---

# 四、注意事项

1. 本仓库不包含完整数据集，需要根据上述数据集说明自行下载并放置到对应目录。
2. 本仓库不建议上传过大的训练权重文件、数据集文件和输出视频文件。
3. 如果在其他电脑上运行，需要检查并修改代码中的本地路径。
4. Windows 系统下路径可以使用 `\`，但建议在 Python 中使用原始字符串格式，例如：

```python
path = r"D:\your\local\path"
```

5. 如果运行任务二时无法读取视频帧或模型权重，请优先检查 `tracker.py` 中的 `model_path` 和 `input_folder` 是否正确。
6. 如果重新训练 YOLOv8 后输出目录与 README 中示例不同，请以本地实际生成的 `best.pt` 路径为准。
7. 任务一和任务三使用的是同一个 Oxford-IIIT Pet Dataset，只是任务一读取分类标签，任务三读取 trimap 分割标注。为了节省空间，也可以只下载一份数据集，并在代码中统一修改数据路径。

---

# 五、项目链接

GitHub 仓库地址：

```text
https://github.com/LuBei1022/CV-Midterm-PJ
```

---

# 六、License

本项目遵循 MIT License。
