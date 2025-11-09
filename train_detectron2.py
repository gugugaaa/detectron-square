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

from detectron2.engine import DefaultTrainer, hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import build_detection_test_loader

DATA_ROOT = dataset_cfg['output_dir']
TRAIN_JSON = os.path.join(DATA_ROOT, "annotations/instances_train.json")
VAL_JSON = os.path.join(DATA_ROOT, "annotations/instances_val.json")
IMAGE_ROOT = DATA_ROOT

# 重新注册数据集
if "square_train" in DatasetCatalog.list():
    DatasetCatalog.remove("square_train")
if "square_val" in DatasetCatalog.list():
    DatasetCatalog.remove("square_val")

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

# 新增：周期性评估与“最佳模型”保存配置
eval_period = int(train_cfg.get('eval_period', 500))
best_metric = train_cfg.get('best_metric', 'segm/AP')   # 对 Mask R-CNN 通常用 segm/AP
best_mode = train_cfg.get('best_mode', 'max')           # AP 越大越好 -> 'max'
best_prefix = train_cfg.get('best_file_prefix', 'model_best')

cfg.TEST.EVAL_PERIOD = eval_period

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

# 确保 BestCheckpointer 在 EvalHook 之后执行
trainer.register_hooks([
    hooks.BestCheckpointer(
        eval_period=eval_period,
        checkpointer=trainer.checkpointer,
        val_metric=best_metric,
        mode=best_mode,
        file_prefix=best_prefix
    )
])

trainer.train()

# 可选：训练结束后再做一次完整验证并打印
evaluator = COCOEvaluator("square_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "square_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))