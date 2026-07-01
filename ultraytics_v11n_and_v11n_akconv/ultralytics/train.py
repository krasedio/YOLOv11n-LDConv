import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 【修正 1】这里必须指向你修改后的模型结构 YAML 文件
    # 这样 YOLO 才会去调用你注册在 tasks.py 里的 AKConv 和 C3k2_AKConv
    model = YOLO(r"C:\Users\Admin\Desktop\ultralytics\ultralytics\cfg\models\11\yolo11n-akconv.yaml") 

    # 如果你想加载预训练权重（虽然科研不建议，但可以加速收敛），取消下面这行的注释
    # model.load('yolo11n.pt') 

    model.train(
        # 【修正 2】data 必须是数据集配置文件（包含 path, train, val, names 的那个）
        data=r"C:\Users\Admin\Desktop\ultralytics\ultralytics\datasets\coco128.yaml", 
        
        task='detect',
        cache=False,
        imgsz=640,
        epochs=100,
        single_cls=False,
        batch=4,           # 你的显存较小，保持 4 是安全的
        close_mosaic=0,
        workers=0,         # Windows 下设为 0 可以避免很多多线程报错
        device='0',
        optimizer='SGD', 
        amp=False,         # 你之前遇到了 CUDA 错误，关闭 amp 是非常明智的
        project='runs/train',
        name='exp_akconv', # 给实验起个明确的名字
    )