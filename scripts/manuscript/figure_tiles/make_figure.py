import math
import os
import glob
import pandas as pd
import cv2


ANNOTATED_IM_DIR = "/home/quentin/Desktop/flatbug_fig/2024-09-27_flatbug/"
RAW_IM_DIR = "/home/quentin/Desktop/flatbug_fig/raw/"
OUR_ANNOT = "./annotated_tiles"
OUR_RAW = "./raw_tiles"


df = pd.read_csv("clean_flatbug_datasets.csv")
df["annotation_file_path"] = [""] * df.shape[0]

for f in glob.glob(os.path.join(ANNOTATED_IM_DIR, "**", "*.jpg"), recursive=True):
    if "00-all" in f:
        continue
    bn = os.path.basename(f)
    for i, (d,t) in enumerate(zip(df.dataset, df._example_name)):
        if t in bn and d in f:
            # df["annotation_file_path"][i] = f
            df.loc[i, "annotation_file_path"] = f


for f in glob.glob(os.path.join(RAW_IM_DIR, "**", "*.jpg"), recursive=True):
    if "00-all" in f:
        continue
    bn = os.path.basename(f)
    for i, (d,t) in enumerate(zip(df.dataset, df._example_name)):
        if t in bn and d in f:
            # df["annotation_file_path"][i] = f
            df.loc[i, "raw_file_path"] = f


os.makedirs(OUR_ANNOT, exist_ok=True)
os.makedirs(OUR_RAW, exist_ok=True)

for i,r in df.iterrows():
    im = cv2.imread(r.annotation_file_path)
    im_raw = cv2.imread(r.raw_file_path)

    h,w = im.shape[0:2]

    if w < 1000:
        left = math.ceil((1000 - w )/2)
        right = math.floor((1000 - w) / 2)
        im = cv2.copyMakeBorder(im, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
        im_raw = cv2.copyMakeBorder(im_raw, 0, 0, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))

    roi = im[r._example_y: r._example_y + 1000, r._example_x : r._example_x + 1000, :]
    roi_raw = im_raw[r._example_y: r._example_y + 1000, r._example_x : r._example_x + 1000, :]
    cv2.imwrite(os.path.join(OUR_ANNOT, f"{r.short_name}.jpg"), roi)
    cv2.imwrite(os.path.join(OUR_RAW, f"{r.short_name}.jpg"), roi_raw)

"""
cd annotated_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done && magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
cd raw_tiles && for i in *.jpg; do magick $i -fill '#0008' -draw 'rectangle  0,100,1000,0' -fill white   -font DejaVu-Sans-Mono-Book -pointsize 64  -annotate +40+70 "$(echo $i| cut -f 1 -d .)" $i; done &&  magick -size 1000x1000 canvas:white AAA.jpg && montage *.jpg -tile 4x6 -geometry 500x500+10+10 tiles.jpeg && cd ..
"""


