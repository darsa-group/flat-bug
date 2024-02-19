import torch
from torchvision.io import read_image

import time

from flat_bug.predictor import Predictor
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

from flat_bug.predictor import *
from flat_bug.yolo_helpers import *


from glob import glob

datasets = {p.removeprefix("s3/") : glob(f'{p}/**.jpg') for p in glob("s3/**")}

# paths = glob("dev/input/**.jpg")[:3]
paths = sorted(glob("s3/CollembolAI/ctrain**.jpg"))[:1]

weights = "model_snapshots/fb_2024-02-19_best.pt"
device = torch.device("cuda:0")
dtype = torch.float16

pred = Predictor(weights, device=device, dtype=dtype)
# pred._model = torch.compile(pred._model,mode="reduce-overhead", dynamic=True)
pred.MIN_MAX_OBJ_SIZE = 8, 2048
pred.MAX_MASK_SIZE = 1024
pred.SCORE_THRESHOLD = 0.3
pred.IOU_THRESHOLD = 0.15
pred.MINIMUM_TILE_OVERLAP = 384
pred.TIME = True

torch.cuda.empty_cache()

start = time.time()
for i in range(len(paths)):
    test = pred.pyramid_predictions(paths[i], scale_increment=1/2, scale_before=1, single_scale=False)
    test.save("dev/input_output", fast=True, crops = False)
    del test
    torch.cuda.empty_cache()

print(f'Average time: {(time.time() - start) / len(paths):.3f}s')