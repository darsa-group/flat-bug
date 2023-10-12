
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
import glob
import os
import pickle

import pandas as pd

# MODEL_PATH = 'runs/detect/train9/weights/best.pt'
MODEL_PATH = 'runs/segment/train12/weights/best.pt'
SOURCE_IMG_DIR = "/home/quentin/Desktop/pitfall_small/smaller"
# SOURCE_IMG_DIR = "/tmp/testd"
TEST_SRC_IMG = "/home/quentin/Desktop/pitfall_small/smaller/pitfall_0.25_2023-07-16_5_W_0m010.jpg"

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.3,
    device="cuda", # or 'cuda:0'
)


result = get_sliced_prediction(TEST_SRC_IMG, detection_model,
                               slice_height=1024,
                               slice_width=1024,
                               overlap_height_ratio=0.1,
                               overlap_width_ratio=0.1,
                                postprocess_type="GREEDYNMM"
                               )



p = predict(
    model_type="yolov8",
    model_path=MODEL_PATH,
    model_device="cuda",
    model_confidence_threshold=0.01,
    source=SOURCE_IMG_DIR,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.1,
    overlap_width_ratio=0.1,
    visual_text_size = 1.0,
    return_dict= True,
    export_pickle = True
)

d = p["export_dir"]
out = os.path.join(d, "results.csv")
summary_out = os.path.join(d, "summary_results.csv")

files = [p for p in sorted(glob.glob(os.path.join(d,"pickles", "*.pickle")))]

all_preds = []
summary_preds = []

for file in files:
    with open(file, 'rb') as f:
        ct = pickle.load(f)
        summary_preds.append({"file": os.path.basename(file), "n": len(ct)})
        for c in ct:
            o = {"file": os.path.basename(file),
                 "x": c.bbox.to_xywh()[0],
                 "y": c.bbox.to_xywh()[1],
                 "w": c.bbox.to_xywh()[2],
                 "h": c.bbox.to_xywh()[3],
                 "area": c.bbox.area}
            all_preds.append(o)

pd.DataFrame(all_preds).to_csv(out,index=False)
pd.DataFrame(summary_preds).to_csv(summary_out,index=False)

#
# result = get_sliced_prediction(
#     "test_positive.jpg",
#     detection_model,
#     slice_height = 1024,
#     slice_width = 1024,
#     overlap_height_ratio = 0.1,
#     overlap_width_ratio = 0.1
# )
#
#
# result.export_visuals(export_dir="demo_data/")