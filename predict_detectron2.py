import os
import numpy as np
import cv2
import torch
import math
import yaml
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config('config.yaml')
dataset_cfg = config['dataset']
predict_cfg = config['predict']

# 1. 配置与模型加载
cfg = get_cfg()  
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))  
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = predict_cfg['score_thresh_test']
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  
cfg.MODEL.WEIGHTS = predict_cfg['weights']

predictor = DefaultPredictor(cfg)  

metadata = MetadataCatalog.get("my_dataset_inference")
metadata.thing_classes = dataset_cfg['class_names']

# 2. 批量读取测试图像
image_dir = predict_cfg['input']
img_paths = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".png"))
]
batch_size = predict_cfg['batch_size']
if batch_size % 2 != 0:
    batch_size += 1
batch = img_paths[:batch_size]

# 3. 推理并可视化
results_list = []
for img_path in batch:
    im = cv2.imread(img_path)
    im = cv2.resize(im, (dataset_cfg['img_size'], dataset_cfg['img_size']))
    outputs = predictor(im)  # 模型推理
    v = Visualizer(im[:, :, ::-1], metadata, scale=1.0)
    out = v.draw_instance_predictions(
        outputs["instances"].to("cpu")
    )
    result = out.get_image()[:, :, ::-1]
    results_list.append(result)

# 4. 马赛克拼接
num_images = len(results_list)
grid_cols = 2
grid_rows = math.ceil(num_images / grid_cols)

# 获取单张图片尺寸
img_h, img_w = results_list[0].shape[:2]

# 创建拼接后的画布
mosaic_h = grid_rows * img_h
mosaic_w = grid_cols * img_w
mosaic = 255 * np.ones((mosaic_h, mosaic_w, 3), dtype=np.uint8)

# 填充图片到马赛克
for idx, img in enumerate(results_list):
    row = idx // grid_cols
    col = idx % grid_cols
    y_start = row * img_h
    x_start = col * img_w
    mosaic[y_start:y_start+img_h, x_start:x_start+img_w] = img

# 保存结果
output_dir = predict_cfg['output']
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "mosaic.png")
cv2.imwrite(output_path, mosaic)
print(f"马赛克拼接结果已保存到: {output_path}")
