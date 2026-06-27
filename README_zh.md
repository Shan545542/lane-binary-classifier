# 车道区域二分类感知模型

本项目是一个面向智能驾驶感知课程作业的轻量级车道区域二分类工程。模型输入车载前视图像中的局部 ROI，输出该区域属于 `in_lane`（车道内）还是 `out_lane`（车道外）。

项目包含：

- TuSimple 车道线标注转换脚本
- 轻量 CNN 二分类模型
- 训练、验证、测试评估脚本
- TensorBoard 训练过程记录
- ONNX 模型导出
- ONNX Runtime 独立推理
- 简单图像增强：水平翻转、亮度扰动、对比度扰动

## 实验结果

最终模型基于 TuSimple 开源车道线数据集转换得到的二分类 ROI 样本训练。

| 数据集 | Accuracy | Precision | Recall | F1 | 样本数 |
|---|---:|---:|---:|---:|---:|
| 验证集 | 0.9085 | 0.8254 | 0.9246 | 0.8722 | 1060 |
| 测试集 | 0.9665 | 0.9301 | 0.9732 | 0.9511 | 1224 |

测试集混淆矩阵：

| 指标 | 数量 | 含义 |
|---|---:|---|
| TP | 399 | 真实为 `in_lane`，预测为 `in_lane` |
| TN | 784 | 真实为 `out_lane`，预测为 `out_lane` |
| FP | 30 | 真实为 `out_lane`，误判为 `in_lane` |
| FN | 11 | 真实为 `in_lane`，误判为 `out_lane` |

ONNX Runtime 推理示例：

- `in_lane` 样本：`probability_in_lane=0.9432`
- `out_lane` 样本：`probability_in_lane=0.0000`

## 项目结构

```text
lane_binary_classifier/
  configs/
    baseline.json
  docs/
    dataset_preparation.md
    experiment_plan.md
    github_upload.md
    model_design.md
  scripts/
    make_toy_dataset.py
    prepare_tusimple_binary.py
  src/lane_binary_classifier/
    data.py
    evaluate.py
    export_onnx.py
    metrics.py
    model.py
    predict.py
    predict_onnx.py
    train.py
  tests/
  requirements.txt
  SUBMISSION.md
```

以下内容不会提交到源码仓库：

```text
data/
outputs/
checkpoints/
*.pt
*.onnx
.venv/
```

## 环境要求

推荐使用：

```text
Python 3.12
```

项目代码要求 Python `>=3.10`。

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

运行依赖包括：

```text
torch
numpy
pillow
tqdm
tensorboard
onnx
onnxruntime
```

## 数据集格式

训练脚本读取 ImageFolder 风格的二分类数据集：

```text
data/tusimple_binary/
  train/
    in_lane/
    out_lane/
  val/
    in_lane/
    out_lane/
  test/
    in_lane/
    out_lane/
```

## TuSimple 数据转换

原始 TuSimple 数据建议放在项目外部，例如：

```text
D:\archive\TUSimple\train_set\
  clips\
  label_data_0313.json
  label_data_0531.json
  label_data_0601.json
```

转换训练集：

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0313.json --split train --output data\tusimple_binary
```

转换验证集：

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0531.json --split val --output data\tusimple_binary
```

转换测试集：

```powershell
python scripts\prepare_tusimple_binary.py --tusimple-root D:\archive\TUSimple\train_set --label-file D:\archive\TUSimple\train_set\label_data_0601.json --split test --output data\tusimple_binary
```

转换脚本会根据 TuSimple 的车道线坐标估计当前车道中心。中心裁剪得到 `in_lane` 样本，左右偏移裁剪得到 `out_lane` 样本。

## 模型训练

```powershell
python -m lane_binary_classifier.train --data-dir data\tusimple_binary --output-dir outputs\tusimple_run --epochs 20 --batch-size 32 --tensorboard
```

训练输出：

```text
outputs/tusimple_run/
  best.pt
  last.pt
  run_config.json
  tensorboard/
```

其中 `best.pt` 根据验证集 F1 保存。本次实验中，最佳模型出现在第 18 个 epoch。

## 查看 TensorBoard

```powershell
tensorboard --logdir outputs\tusimple_run\tensorboard
```

浏览器打开：

```text
http://localhost:6006/
```

记录的指标包括：

- loss
- accuracy
- precision
- recall
- F1

## 测试评估

```powershell
python -m lane_binary_classifier.evaluate --checkpoint outputs\tusimple_run\best.pt --data-dir data\tusimple_binary --split test --output outputs\tusimple_run\test_metrics.json
```

## 导出 ONNX

```powershell
python -m lane_binary_classifier.export_onnx --checkpoint outputs\tusimple_run\best.pt --output outputs\tusimple_run\lane_binary_classifier.onnx
```

## ONNX Runtime 推理

```powershell
python -m lane_binary_classifier.predict_onnx --onnx-model outputs\tusimple_run\lane_binary_classifier.onnx --image data\tusimple_binary\test\in_lane\clips_0601_1494452385593783358_20_center.png
```

示例输出：

```text
class=in_lane probability_in_lane=0.9432 logit=2.8098 provider=CPUExecutionProvider
```

## 数据增强

训练集图像使用简单增强以提高泛化能力：

- 水平翻转
- 亮度扰动
- 对比度扰动

验证集和测试集不使用增强，保证评估结果稳定。

## 提交材料说明

最终提交包包含：

- 源代码压缩包
- PPT 设计文档
- Word 实验报告
- ONNX 模型文件
- 测试指标 JSON
- 训练配置 JSON

具体命令和依赖说明见 `SUBMISSION.md`。
