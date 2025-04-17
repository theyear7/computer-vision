import os
import yaml
import requests
import zipfile
import shutil
from ultralytics import YOLO
import cv2
from pathlib import Path

# 1. 下载并组织数据集
def download_and_prepare_dataset():
    dataset_dir = '../dataset'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 下载 COCO128 数据集
    url = "https://ultralytics.com/assets/coco128.zip"  # COCO128 示例数据集
    zip_path = os.path.join(dataset_dir, "coco128.zip")
    
    if not os.path.exists(zip_path):
        print("Downloading COCO128 dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
    
    # 解压数据集
    extract_dir = os.path.join(dataset_dir, "coco128")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")
    
    # 组织目录结构
    target_dirs = {
        'train/images': os.path.join(dataset_dir, 'train/images'),
        'train/labels': os.path.join(dataset_dir, 'train/labels'),
        'val/images': os.path.join(dataset_dir, 'val/images'),
        'val/labels': os.path.join(dataset_dir, 'val/labels')
    }
    
    for dir_path in target_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 移动文件到目标目录
    source_train_images = os.path.join(extract_dir, 'images/train2017')
    source_train_labels = os.path.join(extract_dir, 'labels/train2017')
    source_val_images = os.path.join(extract_dir, 'images/val2017')
    source_val_labels = os.path.join(extract_dir, 'labels/val2017')
    
    # 复制训练集
    if os.path.exists(source_train_images):
        for file in os.listdir(source_train_images):
            shutil.copy(os.path.join(source_train_images, file), target_dirs['train/images'])
    if os.path.exists(source_train_labels):
        for file in os.listdir(source_train_labels):
            shutil.copy(os.path.join(source_train_labels, file), target_dirs['train/labels'])
    
    # 复制验证集
    if os.path.exists(source_val_images):
        for file in os.listdir(source_val_images):
            shutil.copy(os.path.join(source_val_images, file), target_dirs['val/images'])
    if os.path.exists(source_val_labels):
        for file in os.listdir(source_val_labels):
            shutil.copy(os.path.join(source_val_labels, file), target_dirs['val/labels'])
    
    # 清理临时文件
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return target_dirs

# 2. 创建数据集配置文件
def create_dataset_yaml():
    dataset_config = {
        'train': '../dataset/train/images',
        'val': '../dataset/val/images',
        'nc': 80,  # COCO 数据集有 80 个类别
        'names': [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    }
    
    yaml_path = '../dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f)
    return yaml_path

# 3. 训练模型
def train_model():

    model = YOLO('yolov8n.pt')
    model.train(
        data='../dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='yolo_object_detection'
    )
    return model

# 4. 推理函数
def detect_objects(image_path, model):
    img = cv2.imread(image_path)
    results = model.predict(image_path, conf=0.5)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f'{model.names[cls]} {conf:.2f}'
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    output_path = 'output.jpg'
    cv2.imwrite(output_path, img)
    return img

# 主函数
def main():
    # 下载并准备数据集
    download_and_prepare_dataset()
    
    # 创建 dataset.yaml
    yaml_path = create_dataset_yaml()
    
    # 训练模型
    model = train_model()
    
    # 示例推理
    test_image = 'D:/python/CV/dataset/val/images/000000000357.jpg'
    if os.path.exists(test_image):
        result_img = detect_objects(test_image, model)
        print("检测结果已保存至 output.jpg")
    else:
        print("测试图像不存在，请检查数据集")

if __name__ == '__main__':
    main()
