import math
import os.path

import cv2
import glob
import pandas

all_results = []

valid_classes = {"Rye":"rye", "Grass":"grass", "PureBarley":"barley", "Sample":"barley"}

CROP_DIR = "/home/quentin/Desktop/pollen/pollen_data/all_results/crops/"
CSV_OUT = "/home/quentin/Desktop/pollen/pollen_data/all_results/crops/results.csv"
# im_path = "/home/quentin/Desktop/pollen/pollen_data/all_results/crops/1_a3_p1-20230601_065703_0001_373_1466_0000.png"

for im_path in glob.glob(os.path.join(CROP_DIR, "*.png")):
    im = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)

    im = im[:,:,3]
    cts, _ = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    assert len(cts) == 1, len(cts)
    ct = cts[0]
    area = cv2.contourArea(ct)
    ((x, y), (l, w), angle) = cv2.minAreaRect(ct)
    perimeter = cv2.arcLength(ct, True)
    circularity = 4 * math.pi * (area / (perimeter * perimeter))
    hull = cv2.convexHull(ct)
    convexity = area/ cv2.contourArea(hull)
    basename = os.path.basename(im_path)
    cls = "unknown"
    for k, c in valid_classes.items():
        if basename.startswith(k):
            cls = c

    results = {
        "filename": basename,
        "class": cls,
        "area": area,
               "length": l,
               "width": w,
               "circularity": circularity,
               "convexivity": convexity
               }
    all_results.append(results)
df = pandas.DataFrame(all_results)
df.to_csv(CSV_OUT)


# CSV_OUT = "/home/quentin/Desktop/pollen/pollen_data/all_results/crops/results.csv"
# library(ggplot2)
# library(data.table)
# dt = fread(CSV_OUT)
# ggplot(dt[class!="unknown"], aes(log10(area), circularity, colour=class)) + geom_point(alpha=.3, size=2)
#


# cv2.imshow("ss",im)
# cv2.waitKey(-1)
