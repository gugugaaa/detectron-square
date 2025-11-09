1、新增config.yaml，根据配置生成数据集
dataset:
  output_dir: 'data/coco_squares'
  img_size: 640
  num_squares_per_img_range:
    min: 2
    max: 4
  square_size_range:
    min: 50
    max: 200
  iou_threshold: 0.8
  max_placement_tries: 100
  class_names:
    - 'square'
  splits:
    - name: 'train'
      count: 800
    - name: 'val'
      count: 200
    - name: 'test'
      count: 100
……

2、创建train和predict字段，让.py读取而不是硬编码

PS predict的device不要config

3、predict只需要input不需要root n split，并且有batch参数保存（始终是每行两列），如果batch为奇数就加1.保存到output的文件夹里面。

4、如果train发现没有数据集就调用utils/gen coco dataset来生成