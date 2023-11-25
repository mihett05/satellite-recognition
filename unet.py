import cv2 as cv
import numpy as np
from math import ceil


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
