import os
import cv2
import numpy as np
import random
import json
import yaml
import argparse
from shapely.geometry import Polygon
import time

def load_config(config_path):
    """加载 YAML 配置文件"""
    print(f"Loading configuration from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}. Please create '{config_path}'.")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_iou(box1_coords, box2_coords):
    """
    计算两个多边形（正方形）的相对 IOU。
    使用 (交集面积 / 最小多边形面积) 来检测重叠和包含。
    """
    poly1 = Polygon(box1_coords)
    poly2 = Polygon(box2_coords)
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0

    intersection_area = poly1.intersection(poly2).area
    min_area = min(poly1.area, poly2.area)
    
    return intersection_area / min_area if min_area > 0 else 0.0

def polygon_area(coords):
    """使用 Shoelace 公式计算多边形面积"""
    x = coords[0::2]
    y = coords[1::2]
    return abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0

def generate_image_and_boxes(dataset_cfg):
    """
    生成单个图像及其上的所有正方形。
    返回图像 (np.array) 和一个包含所有正方形原始坐标的列表。
    """
    img_size = dataset_cfg['img_size']
    num_range = dataset_cfg.get('num_squares_per_img_range')
    if num_range:
        min_squares = num_range['min']
        max_squares = num_range['max']
    else:
        legacy = dataset_cfg.get('num_squares_per_img', 1)
        min_squares = max_squares = legacy
    if min_squares > max_squares:
        raise ValueError("num_squares_per_img_range.min must be <= max")
    num_squares = random.randint(min_squares, max_squares)
    size_min = dataset_cfg['square_size_range']['min']
    size_max = dataset_cfg['square_size_range']['max']
    iou_thresh = dataset_cfg['iou_threshold']
    max_tries = dataset_cfg['max_placement_tries']

    # 创建一个白底图像
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    generated_boxes = []  # 存储已生成的正方形 [boxPoints, boxPoints, ...]
    
    for _ in range(num_squares):
        for _ in range(max_tries):
            # 1. 生成随机参数
            size = random.randint(size_min, size_max)
            center_x = random.randint(size // 2, img_size - size // 2)
            center_y = random.randint(size // 2, img_size - size // 2)
            angle = random.uniform(0, 360)
            
            # 2. 创建旋转矩形并获取角点
            rect = ((center_x, center_y), (size, size), angle)
            box_coords = cv2.boxPoints(rect).astype(np.int32)
            
            # 3. 检查边界
            if not np.all((box_coords >= 0) & (box_coords < img_size)):
                continue
            
            # 4. 检查 IOU
            is_valid_iou = True
            for existing_box in generated_boxes:
                iou = calculate_iou(box_coords, existing_box)
                if iou >= iou_thresh:
                    is_valid_iou = False
                    break
            
            if is_valid_iou:
                # 5. 如果所有检查通过，绘制并保存
                cv2.fillPoly(img, [box_coords], (0, 0, 0))  # 填充黑色
                generated_boxes.append(box_coords)
                break  # 成功，跳出重试循环，生成下一个正方形
        else:
            # 如果 max_tries 次尝试都失败了，就放弃这个正方形
            # print(f"Warning: Could not place a square after {max_tries} tries.")
            pass
            
    return img, generated_boxes

def create_coco_dataset(config_path):
    """主函数：加载配置并生成完整的数据集"""
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    # Set seed for reproducibility
    seed = config.get('seed', None)
    if seed is not None:
        print(f"Using random seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)

    dataset_cfg = config['dataset']
    base_dir = dataset_cfg['output_dir']
    
    # 定义 COCO 类别
    categories = [
        {"id": i + 1, "name": name, "supercategory": "shape"}
        for i, name in enumerate(dataset_cfg['class_names'])
    ]
    # 类别 ID 映射 (在这个例子中只有 'square': 1)
    class_to_id = {name: i + 1 for i, name in enumerate(dataset_cfg['class_names'])}
    
    # 跟踪全局唯一的标注 ID
    global_ann_id = 1
    
    start_time = time.time()
    print("Starting COCO dataset generation...")

    # 循环处理 'train', 'val', 'test'
    for split_info in dataset_cfg['splits']:
        split_name = split_info['name']
        num_to_generate = split_info['count']
        
        print(f"\n--- Generating {split_name} split ({num_to_generate} images) ---")
        
        # 准备 COCO 字典
        images_list = []
        annotations_list = []
        coco_output = {
            "info": {"description": f"COCO {split_name} split for {base_dir}"},
            "licenses": [],
            "images": images_list,
            "annotations": annotations_list,
            "categories": categories
        }
        
        # 创建输出目录
        img_dir = os.path.join(base_dir, split_name)
        ann_dir = os.path.join(base_dir, 'annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        generated_count = 0
        attempt_count = 0
        # 设置一个安全阈值，防止无限循环
        max_attempts = max(num_to_generate * 20, 10000) 

        while generated_count < num_to_generate and attempt_count < max_attempts:
            img, raw_boxes = generate_image_and_boxes(dataset_cfg)
            attempt_count += 1

            # 确保至少生成了一个正方形
            if not raw_boxes:
                if attempt_count % 100 == 0:
                    print(f"Warning: Retrying generation (attempt {attempt_count}), "
                          "could not place any valid square.")
                continue
            
            generated_count += 1
            img_id = generated_count  # 在每个 split 内，image_id 从 1 开始
            h, w = img.shape[:2]
            
            # 1. 保存图像文件
            file_name = f'img_{img_id:06d}.jpg'
            img_path = os.path.join(img_dir, file_name)
            cv2.imwrite(img_path, img)
            
            # 2. 创建 COCO 'image' 条目
            images_list.append({
                "id": img_id,
                "width": w,
                "height": h,
                "file_name": file_name
            })
            
            # 3. 为图像中的每个 box 创建 COCO 'annotation' 条目
            for box in raw_boxes:
                coords = box.flatten().astype(float).tolist()
                area = polygon_area(coords)
                
                # 计算 Bounding Box [x_min, y_min, width, height]
                xs = coords[0::2]
                ys = coords[1::2]
                x_min, y_min = min(xs), min(ys)
                bbox_w, bbox_h = max(xs) - x_min, max(ys) - y_min
                bbox = [float(x_min), float(y_min), float(bbox_w), float(bbox_h)]
                
                annotations_list.append({
                    "id": global_ann_id,
                    "image_id": img_id,
                    "category_id": class_to_id['square'], # 在这里使用类别 ID
                    "segmentation": [coords],
                    "area": float(area),
                    "bbox": bbox,
                    "iscrowd": 0
                })
                global_ann_id += 1 # 确保 annotation ID 是全局唯一的
            
            if generated_count % 100 == 0:
                print(f"  ... Generated {generated_count}/{num_to_generate} images for {split_name}")

        if attempt_count >= max_attempts:
             print(f"Warning: Hit max attempts ({max_attempts}) for {split_name}. "
                   f"Generated {generated_count} images instead of {num_to_generate}.")
        
        # 4. 写入 COCO JSON 文件
        json_path = os.path.join(ann_dir, f'instances_{split_name}.json')
        print(f"Writing COCO annotations to: {json_path}")
        with open(json_path, 'w') as jf:
            json.dump(coco_output, jf, ensure_ascii=False, indent=2)

    end_time = time.time()
    print("\n-----------------------------------------")
    print(f"Dataset generation complete in {end_time - start_time:.2f} seconds.")
    print(f"Output directory: {os.path.abspath(base_dir)}")
    print("-----------------------------------------")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate COCO dataset with rotated squares')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file (default: config.yaml)')
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), '..', args.config)
    create_coco_dataset(config_path)