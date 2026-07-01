# pt2torchscript.py
import torch
from ultralytics import YOLO

# 方式1：使用 ultralytics 直接加载（推荐）
model = YOLO("best.pt")  # 自动处理加载
model.export(format="torchscript")  # 直接导出为 TorchScript