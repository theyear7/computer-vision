# YOLOv8 对象检测项目

## 环境要求

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- PyYAML

## 安装

```bash
pip install ultralytics opencv-python pyyaml
```

## 数据集准备

1. 项目会自动创建以下目录结构：

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
```

2. 数据集：

项目将自动下载COCO128数据集到训练集中，然后需要手动选择比例将训练图片和标签分配到验证集中,再进行第二次启动
## 使用方法

1. 运行主程序：

```bash
python main.py
```
2.分配训练集到验证集中，再次运行主程序
3. 训练结果保存在 `runs/detect/yolo_object_detection/`
4. 推理结果保存在 `output.jpg`

## 注意事项

- 确保有足够的GPU/CPU资源进行训练
- 预训练模型会自动下载
- 可通过修改 `dataset.yaml` 调整类别数量和名称
- 训练参数（如epochs、batch size）可在 `train_model()` 函数中调整

