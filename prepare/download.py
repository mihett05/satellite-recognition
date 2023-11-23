import asyncio
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from satellite.utils import deg2num, num2deg, location_to_pixel
from satellite.download import download_bbox_tiles
from satellite.merge import merge
from satellite.overpass import get_building


async def download_bbox(
    bbox: tuple[
        float,
        float,
        float,
        float,
    ],
    output_filename="output.png",
    mask_filename="mask.png",
):  # left, bottom, right, top
    zoom = 18
    bbox = (
        *num2deg(*deg2num(bbox[0], bbox[1], zoom), zoom),
        *num2deg(*deg2num(bbox[2], bbox[3], zoom), zoom),
    )
    grid = await download_bbox_tiles(
        bbox[:2],
        bbox[2:],
        zoom,
    )
    img = merge(grid)
    cv.imwrite(output_filename, img)
    # img = cv.imread(output_filename)

    h, w, _ = img.shape

    buildings = [
        [
            location_to_pixel(point["lat"], point["lon"], bbox, w, h)
            for point in building["geometry"]
        ]
        for building in await get_building(bbox)
        if building["type"] == "way"
    ]

    for building in buildings:
        cv.fillPoly(
            img,
            [np.array([(x, y) for x, y in building], dtype=np.int32)],
            color=(0, 255, 0),
        )

    for y in range(img.shape[0] // 256):
        for x in range(img.shape[1] // 256):
            cv.rectangle(
                img,
                (x * 256, y * 256),
                (x * 256 + 256, y * 256 + 256),
                color=(255, 0, 0),
                thickness=2,
            )

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for building in buildings:
        cv.fillPoly(
            mask,
            [np.array([(x, y) for x, y in building], dtype=np.int32)],
            color=(255,),
        )

    cv.imwrite(mask_filename, mask)
