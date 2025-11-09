import os
import yaml

from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances

from detectron2.engine import DefaultTrainer
from detectron2.engine import hooks as d2_hooks
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo

from utils.gen_coco_dataset import create_coco_dataset


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config('config.yaml')
dataset_cfg = config['dataset']
train_cfg = config['train']

# 准备数据
if not os.path.exists(dataset_cfg['output_dir']):
    print("Dataset not found, generating...")
    create_coco_dataset('config.yaml')

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

# 明确元数据，包含类别与评估类型
MetadataCatalog.get("square_train").set(thing_classes=dataset_cfg['class_names'], evaluator_type="coco")
MetadataCatalog.get("square_val").set(thing_classes=dataset_cfg['class_names'], evaluator_type="coco")


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # 显式评估 bbox 与 segm，确保产生 segm/AP
        return COCOEvaluator(dataset_name, tasks=["bbox", "segm"], distributed=True, output_dir=output_folder)

    def build_hooks(self):
        hook_list = super().build_hooks()
        # 将 BestCheckpointer 放在 EvalHook 之后
        idx = None
        for i, h in enumerate(hook_list):
            if isinstance(h, d2_hooks.EvalHook):
                idx = i
        best_ckpt = d2_hooks.BestCheckpointer(
            eval_period=int(self.cfg.TEST.EVAL_PERIOD),
            checkpointer=self.checkpointer,
            val_metric="segm/AP",
            mode="max",
            file_prefix="model_best"
        )
        if idx is None:
            hook_list.append(best_ckpt)
        else:
            hook_list.insert(idx + 1, best_ckpt)
        return hook_list


# 配置
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(train_cfg['weights']))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(train_cfg['weights'])
cfg.DATASETS.TRAIN = ("square_train",)
cfg.DATASETS.TEST = ("square_val",)
cfg.DATALOADER.NUM_WORKERS = train_cfg['num_workers']
cfg.SOLVER.IMS_PER_BATCH = train_cfg['ims_per_batch']
cfg.SOLVER.BASE_LR = train_cfg['base_lr']
cfg.SOLVER.MAX_ITER = train_cfg['max_iter']
cfg.SOLVER.STEPS = []
cfg.MODEL.ROI_HEADS.NUM_CLASSES = train_cfg['num_classes']
cfg.INPUT.MASK_FORMAT = train_cfg['mask_format']
cfg.MODEL.MASK_ON = True  # 显式开启分割分支（Mask R-CNN）
cfg.OUTPUT_DIR = train_cfg['output_dir']
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# 周期评估与最佳权重
cfg.TEST.EVAL_PERIOD = int(train_cfg.get('eval_period', 500))

# 训练
trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# 训练结束后再评估一次
evaluator = COCOEvaluator("square_val", tasks=["bbox", "segm"], distributed=False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "square_val")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
