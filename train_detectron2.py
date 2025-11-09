import os
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader

DATA_ROOT = "datasets/coco/3squares"
TRAIN_JSON = os.path.join(DATA_ROOT, "instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "instances_val.json")
IMAGE_ROOT = os.path.join("datasets/yolo/3squares/images")  # 所有 split 共用

# 在注册前清除已存在的数据集
if "square_train" in DatasetCatalog.list():
    DatasetCatalog.remove("square_train")
if "square_val" in DatasetCatalog.list():
    DatasetCatalog.remove("square_val")

# 然后重新注册
register_coco_instances("square_train", {}, TRAIN_JSON, os.path.join(IMAGE_ROOT, "train"))
register_coco_instances("square_val", {}, VAL_JSON, os.path.join(IMAGE_ROOT, "val"))
MetadataCatalog.get("square_train").thing_classes = ["square"]

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ("square_val",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.INPUT.MASK_FORMAT = "polygon"
cfg.OUTPUT_DIR = "output/mask_rcnn_square"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("square_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "square_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))