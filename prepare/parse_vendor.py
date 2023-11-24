import os
import shutil
from pathlib import Path

from .convert_yolo import convert_mask_to_yolo, mix_dataset
from .split import split_image


def parse_vendor(convert_yolo: bool = True):
    vendor = Path("vendor")
    for i, image in enumerate(list(os.walk(vendor / "images"))[0][-1]):
        output = str(vendor / "images" / image)
        mask = str(vendor / "masks" / image.replace("image", "mask"))
        dataset_name = f"vendor{i}"
        split_image(output, mask, dataset_name)
        if convert_yolo:
            convert_mask_to_yolo(dataset_name)
            mix_dataset(dataset_name)
