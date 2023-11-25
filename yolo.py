import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics
from math import ceil
from ultralytics import YOLO
from ultralytics.engine.results import Results

yolo = YOLO("vendor_custom_aug.pt")


def train_model():
    yolo.train(data="data.yaml", epochs=10, imgsz=256)


def predict_yolo(model: YOLO, input_filename: str):
    img = cv.imread(input_filename)
    origin_h, origin_w, _ = img.shape
    width = ceil(img.shape[1] / 256) * 256
    height = ceil(img.shape[0] / 256) * 256
    img = np.pad(img, ((0, height - origin_h), (0, width - origin_w), (0, 0)))

    grid = np.zeros((height, width), dtype=bool)

    for y in range(0, (height // 256 - 1) * 256, 128):
        for x in range(0, (width // 256 - 1) * 256, 128):
            if x + 256 >= width or y + 256 >= height:
                break
            tile = img[y : (y + 256), x : (x + 256)]
            result: list[Results] = model(tile, augment=False)
            for r in result:
                if r.masks is not None:
                    for mask in r.masks.data:
                        grid[y : (y + 256), x : (x + 256)] = grid[
                            y : (y + 256), x : (x + 256)
                        ] | np.array(mask.data.cpu(), dtype=bool)
    grid = grid[:origin_h, :origin_w]
    grid = grid.astype(dtype=int)
    return grid
