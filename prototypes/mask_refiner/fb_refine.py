import json
import argparse
import os
import cv2 as cv
import numpy as np


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

def best_yolo_contour(pts, roi_padded, conf_thresh=0.25):
    """
    Choose the best YOLOv8 polygon (on roi_padded) by IoU against `pts` (N,2).
    Uses polygon IoU via shapely if available; otherwise falls back to mask IoU.
    Returns an (M,2) int32 array of the best polygon in ROI coordinates.
    """

    # --- get global YOLOv8 model ---

    H, W = roi_padded.shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return pts.astype(np.int32)

    # --- try shapely for polygon IoU ---
    try:
        from shapely.geometry import Polygon
        from shapely.errors import TopologicalError
        use_shapely = True
    except Exception:
        use_shapely = False

    # helper: polygon IoU with shapely
    def _poly_iou_shapely(a_xy, b_xy):
        try:
            pa = Polygon(a_xy).buffer(0)  # buffer(0) fixes minor self-intersections
            pb = Polygon(b_xy).buffer(0)
            if not pa.is_valid or not pb.is_valid:
                return 0.0
            inter = pa.intersection(pb).area
            union = pa.union(pb).area
            return float(inter / union) if union > 0 else 0.0
        except TopologicalError:
            return 0.0

    # fallback: mask IoU if shapely not available
    def _poly_iou_mask(a_xy, b_xy):
        am = np.zeros((H, W), dtype=np.uint8)
        bm = np.zeros((H, W), dtype=np.uint8)
        a_int = np.round(a_xy).astype(np.int32)
        b_int = np.round(b_xy).astype(np.int32)
        a_int[:, 0] = np.clip(a_int[:, 0], 0, W - 1)
        a_int[:, 1] = np.clip(a_int[:, 1], 0, H - 1)
        b_int[:, 0] = np.clip(b_int[:, 0], 0, W - 1)
        b_int[:, 1] = np.clip(b_int[:, 1], 0, H - 1)
        cv.fillPoly(am, [a_int], 1)
        cv.fillPoly(bm, [b_int], 1)
        inter = np.logical_and(am, bm).sum()
        union = np.logical_or(am, bm).sum()
        return float(inter / union) if union > 0 else 0.0

    # --- run YOLOv8 ---
    try:
        res = yolo(roi_padded, verbose=False)[0]
    except Exception:
        return pts.astype(np.int32)

    # no masks predicted
    if not hasattr(res, "masks") or res.masks is None or getattr(res.masks, "xy", None) is None:
        return pts.astype(np.int32)

    # confidences (aligned with masks)
    try:
        confs = res.boxes.conf.detach().cpu().numpy()
    except Exception:
        confs = np.ones(len(res.masks.xy), dtype=np.float32)

    # choose IoU function
    iou_fn = _poly_iou_shapely if use_shapely else _poly_iou_mask

    best_iou = -1.0
    best_poly = None

    # ground-truth polygon (float)
    gt = pts

    for i, poly in enumerate(res.masks.xy):
        if i < len(confs) and confs[i] < conf_thresh:
            continue
        if poly is None or len(poly) < 3:
            continue

        poly = np.asarray(poly, dtype=np.float32)

        # clip to image bounds to be safe (also helps mask fallback)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)

        iou = iou_fn(gt, poly)
        if iou > best_iou:
            best_iou = iou
            best_poly = poly

    if best_poly is None or best_iou <= 0.0:
        return pts.astype(np.int32)

    return np.round(best_poly).astype(np.int32)


def refine_instance(im, box, cnt, cls):
    h, w = im.shape[:2]
    x1, y1, x2, y2 = box

    # --- 1️⃣ Expand box by 10% ---
    box_w = x2 - x1
    box_h = y2 - y1
    expand_x = int(0.1 * box_w)
    expand_y = int(0.1 * box_h)

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

    refined_candidate = best_yolo_contour(pts, roi_padded)

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
    result_file = "data/metadata_mask-refiner-test_UUID_ChangeThisTEMPORARY.json"
    yolo = YOLO(model_file, "segment", verbose=True)
    refine_file(result_file)
