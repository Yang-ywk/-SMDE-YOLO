import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/home/l611/611/ywk/ultralytics-yolo11/ultralytics/cfg/models/yolo11-mafpn-f.yaml')
    # model.load('yolov11n.pt') # loading pretrain weights
    model.train(data='/home/l611/611/ywk/ultralytics-yolo11/railsem7750-original.yaml',
                cache=False,
                imgsz=640,
                epochs=400,
                batch=32,
                close_mosaic=0,
                workers=8,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/segment',
                name='exp',
                )


