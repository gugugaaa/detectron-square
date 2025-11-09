import os
import yaml
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from utils.gen_coco_dataset import create_coco_dataset

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = load_config('config.yaml')
dataset_cfg = config['dataset']
train_cfg = config['train']

if not os.path.exists(dataset_cfg['output_dir']):
    print("Dataset not found, generating...")
    create_coco_dataset('config.yaml')
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader

DATA_ROOT = dataset_cfg['output_dir']
TRAIN_JSON = os.path.join(DATA_ROOT, "annotations/instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "annotations/instances_val.json")
IMAGE_ROOT = DATA_ROOT  # 所有 split 共用

# 在注册前清除已存在的数据集
if "square_train" in DatasetCatalog.list():
    DatasetCatalog.remove("square_train")
if "square_val" in DatasetCatalog.list():
    DatasetCatalog.remove("square_val")

# 然后重新注册
register_coco_instances("square_train", {}, TRAIN_JSON, os.path.join(IMAGE_ROOT, "train"))
register_coco_instances("square_val", {}, VAL_JSON, os.path.join(IMAGE_ROOT, "val"))
MetadataCatalog.get("square_train").thing_classes = dataset_cfg['class_names']

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(train_cfg['weights']))
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ("square_val",)
cfg.DATALOADER.NUM_WORKERS = train_cfg['num_workers']
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(train_cfg['weights'])
cfg.SOLVER.IMS_PER_BATCH = train_cfg['ims_per_batch']
cfg.SOLVER.BASE_LR = train_cfg['base_lr']
cfg.SOLVER.MAX_ITER = train_cfg['max_iter']
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = train_cfg['num_classes']
cfg.INPUT.MASK_FORMAT = train_cfg['mask_format']
cfg.OUTPUT_DIR = train_cfg['output_dir']
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

evaluator = COCOEvaluator("square_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "square_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))