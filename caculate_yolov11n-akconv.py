from ultralytics import YOLO

def check_yolo_info(model_path):
    try:
        model = YOLO(model_path)
        
        # imgsz=640 是指定计算 GFLOPS 时的输入分辨率
        results = model.info(detailed=False, imgsz=640) 
        
        print("\n" + "="*30)
        print(f"成功加载模型: {model_path}")
        # 输出 Params 和 GFLOPS
        print("="*30)
        
    except Exception as e:
        print(f"模型信息提取失败: {e}")

if __name__ == "__main__":
    MODEL_FILE = "/workspace/boxuan_xing/runs/yolo11n_akconv/weights/best.pt"
    check_yolo_info(MODEL_FILE)