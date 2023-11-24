from unittest.mock import Base
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import sklearn.metrics
from math import ceil
from ultralytics import YOLO
from ultralytics.engine.results import Results

# model = YOLO("yolov8n-seg.pt")
model = YOLO("runs/segment/train17/weights/best.pt")


def train_model():
    model.train(data="data.yaml", epochs=10, imgsz=256)


def predict_yolo(input_filename: str):
    img = cv.imread(input_filename)
    origin_h, origin_w, _ = img.shape
    width = ceil(img.shape[1] / 256) * 256
    height = ceil(img.shape[0] / 256) * 256
    img.resize((height, width, 3))

    grid = np.zeros((height, width))

    for y in range(height // 256 - 1):
        for x in range(width // 256 - 1):
            tile = img[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256]
            result: list[Results] = model(tile)
            for r in result:
                if r.masks is not None:
                    for mask in r.masks.data:
                        grid[
                            y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256
                        ] += np.array(mask.data.cpu())
    grid[grid != 0] = 255
    cv.imwrite("grid.png", grid)


def predict_unet(input_filename: str):
    import segmentation_models_pytorch as smp
    import torch
    from torchvision.io import read_image

    device = "cuda:0" if False and torch.cuda.is_available() else "cpu"
    unet = smp.Unet(activation="sigmoid").to(device)
    unet.load_state_dict(torch.load("unet_best.pt"))

    img = read_image(input_filename)
    print(img.shape)

    _, origin_h, origin_w = img.shape
    width = ceil(img.shape[2] / 256) * 256
    height = ceil(img.shape[1] / 256) * 256
    img = img.view(1, 3, height, width)

    grid = np.zeros((height, width))

    for y in range(height // 256 - 1):
        for x in range(width // 256 - 1):
            tile = img[:, :, y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256]
            with torch.no_grad():
                result = unet(tile.float())[0, 0]
            grid[
                y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256
            ] = result.cpu().numpy()

    grid[grid > 0.5] = 255
    cv.imwrite("grid.png", grid)


def metrics(predict_name: str, true_name: str):
    predict = cv.imread(predict_name, cv.COLOR_BGR2GRAY).flatten()
    true = cv.imread(true_name, cv.COLOR_BGR2GRAY).flatten()
    print(sklearn.metrics.f1_score(true, predict, pos_label=255))


if __name__ == "__main__":
    # import gc

    # gc.collect()

    # torch.cuda.empty_cache()
    # train_model()
    predict_unet("perm_zhelezka_output.png")
    metrics("grid.png", "perm_zhelezka_mask.png")
