from math import ceil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torchvision.transforms import Pad
import segmentation_models_pytorch as smp


device = "cuda:0" if torch.cuda.is_available() else "cpu"


best_model = smp.DeepLabV3Plus(classes=1, activation="sigmoid")
best_model.load_state_dict(torch.load("best_deeplab.pt"))
best_model = best_model.to(device)


def predict_deeplab(image_path: str):
    full_image = read_image(image_path)

    original_height, original_width = full_image.shape[1:]
    desired_height = ceil(original_height / 256.0) * 256
    desired_width = ceil(original_width / 256.0) * 256

    pad = Pad((0, 0, desired_width - original_width, desired_height - original_height))
    full_image = pad(full_image)

    print(
        full_image.shape, desired_height, desired_width, original_height, original_width
    )

    crops = []
    indices = []

    for y in range(desired_height // 256 * 2 - 1):
        for x in range(desired_width // 256 * 2 - 1):
            crops.append(
                full_image[
                    :, y * 128 : y * 128 + 256, x * 128 : x * 128 + 256
                ].unsqueeze(0)
            )
            indices.append((y * 128, y * 128 + 256, x * 128, x * 128 + 256))

    crops = torch.cat(crops, dim=0).float()
    crops_loader = DataLoader(crops, batch_size=1, shuffle=False)

    results = []

    best_model.eval()

    with torch.no_grad():
        for batch in crops_loader:
            x = batch.to(device)
            results.append(best_model(x).cpu())

    result_mask = np.zeros((desired_height, desired_width), dtype=bool)

    for index, result in zip(indices, results):
        top, bottom, left, right = index
        result_mask[top:bottom, left:right] |= result.squeeze().numpy() > 0.5

    result_mask = result_mask[:original_height, :original_width].copy()

    return result_mask.astype(np.uint8) * 255
