import torch
from torchvision.io import read_image

import time

from flat_bug.predictor import Predictor
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

from src.flat_bug.predictor import *
from src.flat_bug.yolo_helpers import *


from glob import glob

datasets = {p.removeprefix("s3/") : glob(f'{p}/**.jpg') for p in glob("s3/**")}

paths = glob("dev/input/**.jpg")[:3]

weights = "model_snapshots/fb_2024-02-09_best.pt"
device = torch.device("cuda:0")
dtype = torch.float16

_model = Predictor(weights, device=device, dtype=dtype)
_model.MINIMUM_TILE_OVERLAP = 384
_model.SCORE_THRESHOLD = 0.2
_model.MAX_MASK_SIZE = 1024
_model.IOU_THRESHOLD = .25
_model.MIN_MAX_OBJ_SIZE = 16, 2048

torch.cuda.empty_cache()

start = time.time()
for i in range(len(paths)):
    img = read_image(paths[i]).to(device, dtype)
    test = _model.pyramid_predictions(img, paths[i], scale_increment=1/2, scale_before=1/2, single_scale=True)
    test.save("dev/output", fast=True, crops = True)
    del test
    torch.cuda.empty_cache()

print(f'Average time: {(time.time() - start) / len(paths):.3f}s')