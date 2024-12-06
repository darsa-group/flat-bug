import os

import torch
from PIL import Image

from flat_bug import download_from_repository
from flat_bug.predictor import Predictor

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))

    image = "cao2022_000178.jpg"
    if not os.path.exists(image):
        download_from_repository(f"fb_yolo/insects/images/val/{image}", image)

    device, dtype = torch.device("cuda:0"), torch.float16
    model = Predictor(
        device=device, 
        dtype=dtype
    )
    model.set_hyperparameters(TIME=True)

    pred = model.pyramid_predictions(
        image, 
        scale_increment=3/4,
        scale_before=1
    ).plot(
        scale=2, 
        confidence=False
    )[1400:2000, 2200:2800]

    Image.fromarray(pred, "RGB").save("example_bbox_vs_contour.png")
