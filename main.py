import asyncio
import os

from prepare.download import download_bbox
from prepare.split import split_image
from prepare.convert_yolo import convert_mask_to_yolo, mix_dataset
from prepare.unite import unite_datasets
from prepare.parse_vendor import parse_vendor

from bboxes import bboxes


async def create_dataset_from_bbox(
    bbox: tuple[float, float, float, float],
    name: str = "tiles",
    convert_yolo: bool = True,
):
    output = f"{name}_output.png"
    mask = f"{name}_mask.png"
    dataset_name = name
    if os.path.exists(os.path.join("datasets", dataset_name)):
        print(f"[ERROR] Dataset '{dataset_name}' exists")
        return
    if not os.path.exists(output) or not os.path.exists(mask):
        await download_bbox(bbox, output, mask)
    split_image(output, mask, dataset_name)
    if convert_yolo:
        convert_mask_to_yolo(dataset_name)
        mix_dataset(dataset_name)


async def download_bboxes(convert_yolo: bool = True):
    await asyncio.gather(
        *[
            create_dataset_from_bbox(bbox, key, convert_yolo=convert_yolo)
            for key, bbox in bboxes.items()
        ]
    )


async def create_yolo():
    await download_bboxes()
    parse_vendor()
    unite_datasets(list(bboxes.keys()), "tiles1", filter=True)
    unite_datasets([f"vendor{i}" for i in range(21)], "vendor", filter=True)
    mix_dataset("tiles1")
    mix_dataset("vendor")
    unite_datasets(["vendor", "tiles1"], "yolo", filter=True)
    mix_dataset("yolo")


async def create_semantic():
    await download_bboxes(convert_yolo=False)
    parse_vendor(convert_yolo=False)
    unite_datasets(
        list(bboxes.keys()),
        "tiles1",
        filter=True,
        convert_yolo=False,
    )
    unite_datasets(
        [f"vendor{i}" for i in range(21)],
        "vendor",
        filter=True,
        convert_yolo=False,
    )
    unite_datasets(
        ["vendor", "tiles1"],
        "tiles2",
        filter=True,
        convert_yolo=False,
    )


asyncio.run(create_semantic())
