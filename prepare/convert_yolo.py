import os
import cv2 as cv
import numpy as np
import shutil
from random import sample
from glob import glob
from pathlib import Path


def convert_mask_to_yolo(dataset_name: str):
    path = Path("datasets") / dataset_name
    files = list(os.walk(path / "masks"))[0][-1]
    h, w = 256, 256
    normal = np.array([w, h])
    for file in files:
        mask = cv.cvtColor(cv.imread(str(path / "masks" / file)), cv.COLOR_BGR2GRAY)
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        rows = []
        for contour in contours:
            row = (contour / normal).flatten().tolist()
            if len(row) >= 6:
                rows.append("1 " + " ".join(map(str, row)))
        shutil.move(path / "images" / file, path / file)
        with open(path / (file.split(".")[0] + ".txt"), "w") as f:
            f.write("\n".join(rows))
    shutil.rmtree(path / "masks")
    shutil.rmtree(path / "images")


def mix_dataset(dataset_name: str):
    path = Path("datasets") / dataset_name
    files = set(glob(str(path / "*.png")))
    val = set(sample(list(files), int(len(files) * 0.2)))
    train = files - val
    os.mkdir(path / "train")
    os.mkdir(path / "val")
    for name, folder in zip(["train", "val"], [train, val]):
        for file in folder:
            p = Path(file)
            base = p.parent
            shutil.move(file, base / name / p.name)
            mask_file = p.name.split(".")[0] + ".txt"
            shutil.move(base / mask_file, base / name / mask_file)
