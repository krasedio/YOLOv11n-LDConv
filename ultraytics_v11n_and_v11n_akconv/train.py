import os
from ultralytics import YOLO

workspace_path = "/workspace/boxuan_xing/yolov11n"
os.makedirs(workspace_path, exist_ok=True)

os.environ['YOLO_CONFIG_DIR'] = workspace_path
os.environ['TMPDIR'] = workspace_path
os.environ['ULTRALYTICS_CONFIG_DIR'] = workspace_path

def train_coco():
    # model = YOLO("/workspace/boxuan_xing/ultralytics/ultralytics/yolo11n.yaml")
    model = YOLO("/workspace/boxuan_xing/ultralytics/ultralytics/cfg/models/11/yolo11.yaml")
    model.train(
        data="coco2017.yaml",
        epochs=500,            # 训练轮次
        imgsz=640,             # 图像尺寸
        batch=256,
        device=[0,1,2,3,4,5,6,7],
        project="/workspace/boxuan_xing",
        workers=16,
        name="runs/yolo11n"
    )

if __name__ == "__main__":
    train_coco()

"/workspace/boxuan_xing/anaconda_data/envs/boxuan/bin/python3 train.py"

