import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
import glob
import os
import json

# MODEL_PATH = 'runs/detect/train9/weights/best.pt'
MODEL_PATH = 'runs/segment/train81/weights/last.pt'

# for s in range(1, 6):
SRC_DIR = "/home/quentin/Downloads/1 Segmentation training dataset"
OUT_DIR = os.path.join(SRC_DIR,"results")

print(SRC_DIR, OUT_DIR)
OUT_JSON = os.path.join(OUT_DIR, "dataset.json")
coco_dict = {
    "categories": [{"id": 1, "name": "insect"}],
    "info": {},
    "licenses": [],
    "images": [],
    "annotations": []
}

os.makedirs(OUT_DIR, exist_ok=True)

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=0.5,
    device="cuda:0",
    # device="cpu",
)
j = 1
for i, src in enumerate(sorted(glob.glob(os.path.join(SRC_DIR, "*.jpg")))):
    # src = "/home/quentin/Desktop/pitfall_small/smaller/pitfall_0.25_2023-07-16_5_W_0m010.jpg"
    im = cv2.imread(src)
    h, w, _ = im.shape
    print(i, src)
    coco_dict["images"].append({
        "id": i +1,
        "file_name": os.path.basename(src),
        "height": h,
        "width": w
    })

    result = get_sliced_prediction(src, detection_model,
                                   slice_height=1024,
                                   slice_width=1024,
                                   overlap_height_ratio=0.25,
                                   overlap_width_ratio=0.25,
                                   postprocess_type="GREEDYNMM",
                                   )

    result.export_visuals(OUT_DIR, file_name=os.path.splitext(os.path.basename(src))[0], text_size=1., rect_th=2)
    ann = result.to_coco_predictions()
    for a in ann:
        a["id"] = j
        a["image_id"] = i +1
        a["category_id"] = 1 # fixme
        j += 1

    coco_dict["annotations"].extend(ann)

with open(OUT_JSON, "w") as f:
    json.dump(coco_dict, f)
