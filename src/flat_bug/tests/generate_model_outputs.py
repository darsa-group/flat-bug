import argparse

import torch
from torchvision.io import read_image

from flat_bug.tests.test_predictor import DummyModel
from flat_bug.tests.test_predictor import ASSET_DIR

# Command I used: python3 src/flat_bug/tests/generate_model_outputs.py --model model_snapshots/fb_2024-03-18_large_best.pt --image src/flat_bug/tests/assets/ALUS_Non-miteArachnids_Unknown_2020_11_03_4545.jpg --type both

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="The model to test")
    parser.add_argument("--image", type=str, required=True, help="The image to test")
    parser.add_argument("--assets", type=str, default=ASSET_DIR, help="The directory to save the assets")
    parser.add_argument("--type", type=str, choices=["single_scale", "pyramid", "both"], required=True)
    parser.add_argument("--device", type=str, default="cuda:0", help="The device to use for inference")
    parser.add_argument("--dtype", type=str, default="float16", help="The dtype to use for inference")

    args = parser.parse_args()

    image = read_image(args.image).to(torch.device(args.device), dtype=getattr(torch, args.dtype))
    test = DummyModel("single_scale", args.assets) # The type doesn't matter here
    match args.type:
        case "single_scale":
            test.generate_single_scale_files(args.model, image)
        case "pyramid":
            test.generate_pyramid_files(args.model, image, args.image)
        case "both":
            test.generate_single_scale_files(args.model, image)
            test.generate_pyramid_files(args.model, image, args.image)
        case _:
            raise ValueError(f"Invalid type {args.type}")
    