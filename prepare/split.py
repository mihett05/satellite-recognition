import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def split_image(img_path: str, mask_path: str, dataset_name: str):
    img = cv.imread(img_path)
    mask = cv.cvtColor(cv.imread(mask_path), cv.COLOR_BGR2GRAY)

    h, w, _ = img.shape

    base_path = Path("datasets") / dataset_name
    os.mkdir(base_path)
    os.mkdir(base_path / "images")
    os.mkdir(base_path / "masks")

    counter = 0

    for y in range(h // 256 - 1):
        for x in range(w // 256 - 1):
            tile = img[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256]
            tile_mask = mask[y * 256 : (y + 1) * 256, x * 256 : (x + 1) * 256]

            mask_result = np.zeros(tile_mask.shape[:2])
            contours, _ = cv.findContours(tile_mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            cv.drawContours(mask_result, contours, -1, (255, 0, 0), thickness=-1)

            cv.imwrite(str(base_path / "images" / f"{counter:>04}.png"), tile)
            cv.imwrite(str(base_path / "masks" / f"{counter:>04}.png"), mask_result)

            counter += 1
