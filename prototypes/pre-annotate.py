# This is a sample Python script.
import glob
import os.path

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import cv2
import numpy as np
import json

def annotations(file_path):
    # Use a breakpoint in the code line below to debug your script.
    im = cv2.imread(file_path)


    grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # cv2.namedWindow("test", 0)
    grey = cv2.erode(grey,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10)))
    grey = cv2.dilate(grey,
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
    grey  = cv2.medianBlur(grey, 11)

    grey_bg = cv2.blur(grey, (751, 751))
    grey = cv2.subtract(grey_bg, grey)
    grey = cv2.dilate(grey,
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    _, grey = cv2.threshold(grey, 20, 255, cv2.THRESH_BINARY)
    conts, _ = cv2.findContours(grey, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in conts:
        x,y,w,h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 2000 and max(w, h) < 4000:
            peri = cv2.arcLength(c, True)
            c = cv2.approxPolyDP(c, 0.002 * peri, True)
            out.append(c)

    conts = out
    im = cv2.drawContours(im, conts, -1, (255,0,0), 3, cv2.LINE_AA)
    small = cv2.resize(im, (0, 0), fx=0.25, fy=0.25)
    cv2.imwrite(os.path.join("/tmp/pitfall_annotations", "ann_" + os.path.basename(file_path)), small)
    return conts

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_paths = [s  for s in sorted(glob.glob("/home/quentin/Desktop/pitfall/*.jpg"))]


    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "insect"}]  # Your category
    }

    annotation_id = 0

    for entry in file_paths:
        print(entry)
        image_path = entry
        contours = annotations(entry)

        # Load the image
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # Populate images
        image_info = {
            "id": len(coco_data["images"]) + 1,
            "file_name": image_path.split("/")[-1],
            "height": height,
            "width": width
        }
        coco_data["images"].append(image_info)

        # Populate annotations for each contour
        for contour in contours:
            annotation_id += 1
            segmentation = [contour.flatten().tolist()]

            # Calculate the bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [x, y, w, h]

            annotation_info = {
                "id": annotation_id,
                "image_id": image_info["id"],
                "category_id": 1,  # Category ID for pollen
                "segmentation": segmentation,
                "area": cv2.contourArea(contour),
                "bbox": bbox,
                "iscrowd": 0  # Assuming all instances are not crowded
            }
            coco_data["annotations"].append(annotation_info)

    # Write COCO JSON file
    with open("coco_dataset.json", "w") as json_file:
        json.dump(coco_data, json_file)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
