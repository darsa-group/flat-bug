import json
import argparse
import os

import cv2
import cv2 as cv
import numpy as np
from torchgen.gen_functionalization_type import return_from_mutable_noop_redispatch


# fixme, resume should continue on the same "run folder"
def main():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    args_parse.add_argument("-d", "--data-dir", dest="data_dir",
                            help="The directory containing the prepared data (i.e., the output of  `fb_prepare.py`",
                            type=str)

    args_parse.add_argument("-c", "--config-file", dest="config_file",
                            help="A YAML-formatted config file that overrides the default training meta-parameters",
                            default=None)
    args_parse.add_argument("-r", "--resume", dest="resume",
                            help="resume training",
                            action='store_true')

    args, extra = args_parse.parse_known_args()

# def best_new_contour(cnt):



def _iou_from_masks(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def yolo_ensemble_contour(image_bgr, cnt, conf_threshold=0.25, iou_threshold=0.1, mask_threshold=1):


    # get global YOLOv8 model

    H, W = image_bgr.shape[:2]
    accum = np.zeros((H, W), dtype=np.float32)
    original_mask = np.zeros_like(accum).astype(np.uint8)
    cv.fillPoly(original_mask, [cnt.astype(np.int32)], 1)
    n=4
    for i in range(n):
        print(i)
        if i ==0:
            image_in = np.copy(image_bgr)
        elif i == 1:
            image_in =  cv.rotate(image_bgr, cv.ROTATE_90_CLOCKWISE)
        elif i == 2:
            image_in = cv.flip(image_bgr, 0)
        elif i == 3:
            image_in = cv.flip(cv.rotate(image_bgr, cv.ROTATE_90_CLOCKWISE), 0)
        else:
            raise ValueError("aug_idx must be in {0,1,2,3}")

        # image_in =cv.medianBlur(image_bgr,i*2+1)
        res = yolo(image_in, verbose=False)[0]

        # no masks predicted this run
        if not hasattr(res, "masks") or res.masks is None or getattr(res.masks, "data", None) is None:
            continue

        confs = res.boxes.conf.detach().cpu().numpy()
        masks_t = res.masks.data  # torch.Tensor [num, h, w]
        num_masks = masks_t.shape[0]

        valid_masks = []
        mask_areas = []

        for j in range(num_masks):
            if confs is not None and j < len(confs) and confs[j] < conf_threshold:
                continue

            m = masks_t[j].detach().cpu().numpy().astype(np.float32)  # float mask (h, w) in [0,1]

            iou = _iou_from_masks(m, original_mask)
            print(j, iou)
            if iou < iou_threshold :
                continue
            valid_masks.append(m)
            mask_areas.append(np.sum(m))

        if len(mask_areas) == 0:
            continue

        m = valid_masks[np.argmax(mask_areas)]

        if i == 0:
            m = m
        elif i == 1:
            m = cv.rotate(m, cv.ROTATE_90_COUNTERCLOCKWISE)
        elif i == 2:
            m = cv.flip(m, 0)
        elif i == 3:
            m = cv.rotate(cv.flip(m,0), cv.ROTATE_90_COUNTERCLOCKWISE)
        else:
            raise ValueError("aug_idx must be in {0,1,2,3}")
        print (i,m.shape)
        accum += m

    frac = accum / float(n)
    cv2.imshow("test2", frac)
    final_mask = (frac >= mask_threshold).astype(np.uint8) * 255
    if np.count_nonzero(final_mask) == 0:
        return cnt
    # find largest contour
    cnts, _ = cv.findContours(final_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    best = max(cnts, key=cv.contourArea)  # largest by area
    best = cv.approxPolyDP(best, 0.0005 * cv.arcLength(best, True), True)
    return best


def refine_instance(im, box, cnt, cls):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = box

    # --- 1️⃣ Expand box by 10% ---
    box_w = x2 - x1
    box_h = y2 - y1
    expand_x = int(0.2 * box_w)
    expand_y = int(0.2 * box_h)

    x1_exp, y1_exp = x1 - expand_x, y1 - expand_y
    x2_exp, y2_exp = x2 + expand_x, y2 + expand_y

    # --- 2️⃣ Clip to image bounds ---
    x1_clip, y1_clip = max(x1_exp, 0), max(y1_exp, 0)
    x2_clip, y2_clip = min(x2_exp, w), min(y2_exp, h)

    # --- 3️⃣ Extract valid region ---
    roi = im[y1_clip:y2_clip, x1_clip:x2_clip]

    # White background of expanded size
    roi_h, roi_w = y2_exp - y1_exp, x2_exp - x1_exp
    white_bg = np.ones((roi_h, roi_w, 3), dtype=np.uint8) * 255

    y_offset = y1_clip - y1_exp
    x_offset = x1_clip - x1_exp
    white_bg[y_offset:y_offset + roi.shape[0], x_offset:x_offset + roi.shape[1]] = roi

    # --- 4️⃣ Contour alignment (relative to expanded ROI) ---
    xs, ys = cnt[0], cnt[1]
    # assert xs.shape == ys.shape, "xs and ys must have same length"
    cnt_np = np.stack([xs, ys], axis=1).astype(np.float32)

    cnt_np[:, 0] -= x1_exp
    cnt_np[:, 1] -= y1_exp

    # --- 5️⃣ Scale so that longest side = 1024 ---
    long_side = max(roi_w, roi_h)
    scale = 1024 / long_side
    new_w = int(roi_w * scale)
    new_h = int(roi_h * scale)

    roi_scaled = cv.resize(white_bg, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    cnt_scaled = cnt_np * scale

    # --- 6️⃣ Pad to 1024×1024 with white margins ---
    pad_x = (1024 - new_w) // 2
    pad_y = (1024 - new_h) // 2

    roi_padded = np.ones((1024, 1024, 3), dtype=np.uint8) * 255
    roi_padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = roi_scaled

    # Adjust contour for padding (relative to final ROI)
    cnt_final = cnt_scaled + np.array([pad_x, pad_y], dtype=np.float32)

    # --- 7️⃣ Keep mapping info to reconstruct original coordinates ---
    transform_info = {
        "x1_exp": x1_exp,
        "y1_exp": y1_exp,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y
    }


    pts = np.array(cnt_final, dtype=np.int32).reshape(-1, 2)

    refined_candidate = yolo_ensemble_contour( roi_padded, pts)

    cv.polylines(roi_padded, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
    cv.polylines(roi_padded, [refined_candidate], isClosed=True, color=(255, 0, 0), thickness=2)
    cv.imshow("test", roi_padded)
    cv.waitKey(-1)



def refine_file(path):
    with open(path, 'r') as f:
        data = json.load(f)

    dir_name = os.path.dirname(path)
    parent_image_rel_path = data["image_path"]
    # parent_image_abs_path = os.path.join(dir_name, parent_image_rel_path)
    parent_image_abs_path = os.path.join( parent_image_rel_path)
    assert os.path.isfile(parent_image_abs_path), parent_image_abs_path

    im = cv.imread(parent_image_abs_path)


    for box, cnt, cls in zip(data["boxes"], data["contours"], data["classes"]):
        refine_instance(im, box,  cnt, cls)


if __name__ == "__main__":
    # main()
    from ultralytics import YOLO
    from ultralytics.engine.results import Results
    model_file = "flat_bug_S.pt"
    result_file = "data/metadata_mask-refiner-test3_UUID_ChangeThisTEMPORARY.json"
    yolo = YOLO(model_file, "segment", verbose=True)
    refine_file(result_file)
