import glob
import json
import os
import shutil

from flat_bug.coco_utils import fb_to_coco

pattern = "/home/quentin/Desktop/nhm-toworkon/**/*.json"
out = "/tmp/nhm_400/"
pred_coco= {}
for p in glob.glob(pattern):
    with open(p, 'r') as f:
        d = json.load(f)
    fb_to_coco(d, pred_coco)
    bn = os.path.basename(os.path.dirname(p))
    im_f = "/home/quentin/Desktop/flat-bug-sorted-data/pre-pro/NHM-beetles/" + bn + ".jpg"
    shutil.copy(im_f, out + bn + ".jpg")

with open("/tmp/compiled_nhm_coco.json", "w") as f:
    json.dump(pred_coco, f)