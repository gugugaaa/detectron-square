# Detectron2 实例分割完整指南

## 目录
- [数据集注册](#数据集注册)
- [配置与训练](#配置与训练)
- [推理](#推理)

---

## 数据集注册

### 注册COCO格式数据集

```python
from detectron2.data.datasets import register_coco_instances

# 注册 COCO 实例分割数据集
register_coco_instances(
    "square_train",                # 数据集名
    {},                            # 可选元数据字典
    "path/to/annotations.json",    # COCO JSON 标注文件
    "path/to/image_directory"      # 图像目录
)
```

### COCO数据集多边形标注格式

**segmentation 字段说明：**
- 类型：由多个顶点数组组成的列表
- 格式：`[x1, y1, x2, y2, …, xn, yn]`
- 用途：描绘对象轮廓

**支持复杂形状：**
- 多个不连续区域：在 "segmentation" 中包含多个顶点数组
- 带孔多边形：使用多个数组表示外轮廓和内孔

---

## 配置与训练

### 完整训练代码

```python
import numpy as np
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

# 加载预训练配置
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# 数据集配置
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2

# 模型和权重配置
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 只有一个类别

# 训练参数
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000

# 输入格式
cfg.INPUT.MASK_FORMAT = "polygon"

# 启动训练
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 关键配置说明

| 配置项 | 说明 |
|--------|------|
| 预训练模型 | 使用 Mask R-CNN R50-FPN 作为基础 |
| `DATASETS.TRAIN` | 指定训练集为 `"square_train"` |
| `MODEL.ROI_HEADS.NUM_CLASSES` | 设置为 1（仅正方形类别） |
| `INPUT.MASK_FORMAT` | 采用多边形分割格式 |
| `SOLVER.BASE_LR` | 0.00025 的学习率适应小数据集 |
| `SOLVER.IMS_PER_BATCH` | 批大小为 2 |

---

## 推理

### 核心API说明

- **`get_cfg()`**：创建并返回默认配置对象，用于加载模型配置文件并进行参数合并

- **`DefaultPredictor(cfg)`**：基于配置构造推理器，内部完成模型加载、预处理、后处理等操作

- **`MetadataCatalog.get(name)`**：获取数据集元信息（类别名称、颜色等），用于可视化初始化

- **`Visualizer(image, metadata, scale)`**：将推理结果绘制到原图上，支持实例分割、检测框等可视化

PS
默认只是保存model final，需要手动注册评估和保存best model的BestCheckpointer