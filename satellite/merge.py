import cv2 as cv
import numpy as np


def merge(i_grid: list[list[bytes]]) -> np.ndarray:
    grid = None
    for y, r in enumerate(i_grid):
        row = None
        for x, tile in enumerate(r):
            nparr = np.frombuffer(tile, dtype=np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
            if row is None:
                row = img
            else:
                row = np.hstack((row, img))
        if grid is None:
            grid = row
        elif row is not None:
            grid = np.vstack((grid, row))
    if grid is None:
        raise ValueError("'grid' is None")
    return grid
