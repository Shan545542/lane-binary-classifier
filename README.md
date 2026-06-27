# Lane Binary Classifier

轻量级车道区域二分类模型：输入车载前视图像的局部 ROI，输出 `in_lane` 或 `out_lane`。项目面向智能驾驶课程作业，同时保持标准 GitHub 项目结构，方便后续补充实验报告、PPT、ONNX 导出和真实数据集训练结果。

## Features

- 轻量 CNN 二分类模型，适合课程级快速训练和解释
- 支持 `ImageFolder` 风格数据集：`train/val` 下各有 `in_lane`、`out_lane`
- 提供 TuSimple 标注转换脚本，可从开源车道线标注生成二分类 ROI 样本
- 支持随机翻转、亮度增强、准确率、精确率、召回率、F1 指标
- 支持 TensorBoard 记录训练过程
- 支持 PyTorch checkpoint 推理、ONNX 导出和 ONNX Runtime 独立推理
- 提供 toy 数据集生成脚本，用于无真实数据时快速跑通流程

## Project Layout

```text
lane_binary_classifier/
  src/lane_binary_classifier/
    data.py            # 数据集、增强、DataLoader
    metrics.py         # 二分类指标
    model.py           # 轻量 CNN 模型
    train.py           # 训练入口
    predict.py         # 单图推理
    export_onnx.py     # ONNX 导出
    predict_onnx.py    # ONNX Runtime 推理
  scripts/
    make_toy_dataset.py
    prepare_tusimple_binary.py
  tests/
  requirements.txt
  pyproject.toml
```

更多说明：

- 数据集转换：`docs/dataset_preparation.md`
- GitHub 上传：`docs/github_upload.md`
- 模型设计：`docs/model_design.md`
- 实验计划：`docs/experiment_plan.md`

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

如果使用 GPU，请按 PyTorch 官网说明安装匹配 CUDA 的 `torch` 版本。

## Quick Start With Toy Data

先生成一个可训练的小型合成数据集：

```bash
python scripts/make_toy_dataset.py --output data/toy_lane --train-size 400 --val-size 80
```

训练模型：

```bash
python -m lane_binary_classifier.train ^
  --data-dir data/toy_lane ^
  --output-dir outputs/toy_run ^
  --epochs 8 ^
  --batch-size 32 ^
  --tensorboard
```

单图推理：

```bash
python -m lane_binary_classifier.predict ^
  --checkpoint outputs/toy_run/best.pt ^
  --image data/toy_lane/val/in_lane/0000.png
```

导出 ONNX：

```bash
python -m lane_binary_classifier.export_onnx ^
  --checkpoint outputs/toy_run/best.pt ^
  --output outputs/toy_run/lane_binary_classifier.onnx
```

ONNX Runtime 推理：

```bash
python -m lane_binary_classifier.predict_onnx ^
  --onnx-model outputs/toy_run/lane_binary_classifier.onnx ^
  --image data/toy_lane/val/in_lane/0000.png
```

## Dataset Format

训练脚本默认读取下面的目录结构：

```text
data/my_lane_binary/
  train/
    in_lane/
      xxx.png
    out_lane/
      xxx.png
  val/
    in_lane/
      xxx.png
    out_lane/
      xxx.png
```

类别含义：

- `in_lane`: ROI 中车辆当前位置位于可行驶车道区域内
- `out_lane`: ROI 中车辆当前位置偏离当前车道区域，或 ROI 与目标车道中心明显错位

## Prepare TuSimple Binary Samples

下载并解压 TuSimple lane detection 数据集后，可用脚本从官方 JSON line 标注生成 ROI 二分类样本：

```bash
python scripts/prepare_tusimple_binary.py ^
  --tusimple-root D:\datasets\tusimple ^
  --label-file D:\datasets\tusimple\train_set\label_data_0313.json ^
  --split train ^
  --output data/tusimple_binary
```

脚本会基于靠近车辆前方的参考横线估计左右车道线，生成一个正样本 crop 和若干左右偏移的负样本 crop。真实训练时建议把多个 label 文件都转换进同一个输出目录，并手动抽查部分样本质量。

## Model

当前模型为 `LaneBinaryNet`：

- 4 个卷积块：Conv2d + BatchNorm + ReLU + MaxPool
- Adaptive Average Pooling 汇聚空间特征
- Dropout + Linear 输出单个 logit
- 损失函数：`BCEWithLogitsLoss`
- 阈值：`sigmoid(logit) >= 0.5` 判为 `in_lane`

该结构参数量小，训练速度快，适合作业中验证“基于局部 ROI 判断车道内/车道外”的基础可行性。

## Next Milestones

- 用 TuSimple/CULane 子集生成真实训练集
- 记录 TensorBoard 曲线截图
- 导出 ONNX 并用 ONNX Runtime 做一次推理验证
- 整理实验报告和 PPT：数据来源、模型结构、训练设置、指标、误差分析
