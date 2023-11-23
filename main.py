import asyncio
import os

from prepare.download import download_bbox
from prepare.split import split_image
from prepare.convert_yolo import convert_mask_to_yolo, mix_dataset
from prepare.unite import unite_datasets
from prepare.parse_vendor import parse_vendor

from bboxes import bboxes


async def create_dataset_from_bbox(
    bbox: tuple[float, float, float, float], name: str = "tiles"
):
    output = f"{name}_output.png"
    mask = f"{name}_mask.png"
    dataset_name = name
    if os.path.exists(os.path.join("datasets", dataset_name)):
        print(f"[ERROR] Dataset '{dataset_name}' exists")
        return
    await download_bbox(bbox, output, mask)
    split_image(output, mask, dataset_name)
    convert_mask_to_yolo(dataset_name)
    mix_dataset(dataset_name)


async def download_bboxes():
    await asyncio.gather(
        *[create_dataset_from_bbox(bbox, key) for key, bbox in bboxes.items()]
    )


async def main():
    await download_bboxes()
    parse_vendor()
    unite_datasets(list(bboxes.keys()), "tiles1")
    unite_datasets([f"vendor{1}" for i in range(21)], "vendor")
    unite_datasets(["vendor", "tiles1"], "tiles2")
    mix_dataset("tiles2")


asyncio.run(main())
