import warnings
warnings.filterwarnings('ignore')
import os
from ultralytics import YOLO


def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = ('runs/segment/exp/weights/best.pt')
    model = YOLO(model_path) # 选择训练好的权重路径
    result = model.val(data='/home/l611/611/ywk/ultralytics-yolo11/railsem7750-original.yaml',
                        split='test', # split可以选择train、val、test
                        imgsz=640,
                        batch=1,
                        # iou=0.7,
                        # rect=False,
                        save_json=True, # if you need to cal coco metrice
                        project='runs/test',
                        name='exp',
                        )
