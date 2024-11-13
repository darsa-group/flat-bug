
import argparse
import json
import os
from glob import glob

from tqdm import tqdm

from flat_bug import logger
from flat_bug.coco_utils import fb_to_coco, filter_coco, split_annotations
from flat_bug.config import DEFAULT_CFG, read_cfg
from flat_bug.eval_utils import compare_groups


def load_json(file : str):
    with open(file, "r") as f:
        return json.load(f)

# Wrapper function to call compare_groups with a single parameter dictionary for multiprocessing
def process_image(params):
    return compare_groups(**params)

def main():
    # # Development defaults
    # predictions = "dev/**/**.json"
    # ground_truth = "s3/CollembolAI/instances_default.json"
    # image_directory = "s3/CollembolAI"
    # output_directory = "dev/eval"
    # iou_match_threshold = 0.1

    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('-p', '--predictions', type=str, help='Path or pattern to the predictions files', required=True)
    parser.add_argument('-g', '--ground_truth', type=str, help='Path to the ground truth file', required=True)
    parser.add_argument('-I', '--image_directory', type=str, help='Path to the image directory', required=True)
    parser.add_argument('-o', '--output_directory', type=str, help='Path to the output directory', required=True)
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=False)
    parser.add_argument('-P', '--plot', action="store_true", help='Plot the matches and the IoU matrix')
    parser.add_argument('-b', '--no_boxes', action="store_false", help='Do not plot the bounding boxes')
    parser.add_argument('-c', '--coco_predictions', action="store_true", help='Whether the predictions are already in a COCO format (legacy)')
    parser.add_argument('-s', '--scale', type=float, default=1, help='Scale of the output images. Defaults to 1. Lower is faster.')
    parser.add_argument('-n', type=int, default=-1, help='Number of images to process. Defaults to -1 (all images)')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers to use for the evaluation. Defaults to 8.')
    parser.add_argument('--combine', action="store_true", help='Combine the results into a single CSV file')
    
    args = parser.parse_args()

    if args.config is not None:
        config = read_cfg(args.config)
    else:
        config = DEFAULT_CFG
    min_size = config["MIN_MAX_OBJ_SIZE"][0]
    confidence_threshold = config["SCORE_THRESHOLD"]
    iou_match_threshold = config["IOU_THRESHOLD"]

    if args.coco_predictions:
        pred_coco = load_json(args.predictions)
    else:
        files = sorted(glob(args.predictions, recursive=True))
        flat_bug_predictions = [load_json(p) for p in files]
        pred_coco = {}
        for d in flat_bug_predictions:
            fb_to_coco(d, pred_coco)
    pred_coco = filter_coco(pred_coco, confidence=confidence_threshold, area=min_size, verbose=False)

    if not os.path.exists(args.ground_truth):
        raise ValueError(f'Ground truth file not found: {args.ground_truth}')
    gt_coco = load_json(args.ground_truth)
    gt_coco = filter_coco(gt_coco, area=min_size)
    gt_annotations, pred_annotations = split_annotations(gt_coco), split_annotations(pred_coco)

    # Find the differences between which images are in the ground truth and which are in the predictions
    gt_keys = set(gt_annotations.keys())
    pred_keys = set(pred_annotations.keys())
    gt_diff_keys = sorted(list(gt_keys.difference(pred_keys)))
    pred_diff_keys = sorted(list(pred_keys.difference(gt_keys)))
    shared_keys = gt_keys.intersection(pred_keys)
    if len(gt_diff_keys) > 0:
        show = min(2, len(gt_diff_keys))
        missing_gt_diff_formatted = ', '.join(['"' + str(i) + '"' for i in gt_diff_keys[:show]])
        logger.info(
            f'Ground truth has {len(gt_diff_keys)} images that are not in the predictions:'
            f'[{missing_gt_diff_formatted}{", ..." if len(gt_diff_keys) > show else ""}] and {len(gt_diff_keys) - show} more'
        )
    if len(pred_diff_keys) > 0:
        show = min(2, len(pred_diff_keys))
        missing_pred_diff_formatted = ', '.join(['"' + str(i) + '"' for i in pred_diff_keys[:show]])
        logger.info(
            f'Predictions has {len(pred_diff_keys)} images that are not in the ground truth:'
            f'[{missing_pred_diff_formatted} {", ..." if len(pred_diff_keys) > show else ""}] and {len(pred_diff_keys) - show} more'
        )
    if len(shared_keys) == 0:
        raise ValueError(f'No images in common between the ground truth and the predictions')

    shared_keys = sorted(shared_keys)
    if args.n != -1:
        logger.info(f'Skipping the evaluation of {len(shared_keys) - args.n} images')
        shared_keys = shared_keys[:args.n]
    if len(shared_keys) == 0:
        raise ValueError(f'No images to evaluate')
    if len(shared_keys) < args.workers:
        args.workers = min(args.workers, len(shared_keys))
        logger.info(f"Warning: More workers than images, reducing the number of workers to {args.workers}")
    
    result_files = []
    
    if args.workers <= 1:
        for image in tqdm(shared_keys, desc="Evaluating images", dynamic_ncols=True):
            result_files += [process_image(image)]
    else:
        from multiprocessing import Pool
        pool = Pool(args.workers)
        all_params = []
        for image in tqdm(shared_keys, desc="Generating parameters for multiprocessing", dynamic_ncols=True, leave=False):
            this_params = {
                "group1"            : gt_annotations[image], 
                "group2"            : pred_annotations[image], 
                "group_labels"      : ["Ground Truth", "Predictions"],
                "image_path"        : f'{args.image_directory}{os.sep}{image}', 
                "output_identifier" : os.path.splitext(image)[0], 
                "plot"              : args.plot,
                "plot_scale"        : args.scale,
                "plot_boxes"        : args.no_boxes,
                "output_directory"  : args.output_directory,
                "threshold"         : iou_match_threshold
            }
            all_params.append(this_params)
        for matches in tqdm(pool.imap_unordered(process_image, all_params), total=len(shared_keys), desc="Evaluating images", dynamic_ncols=True):
            result_files += [matches]
        pool.close()
        pool.join()

    if args.combine:
        import pandas as pd

        # Add the basepath (without .csv) as a new column before concatenating
        def read_and_add_new_column(f):
            df = pd.read_csv(f, sep=";")
            df.insert(0, "image", os.path.splitext(os.path.basename(f))[0])
            return df
        combined_result = pd.concat([read_and_add_new_column(f) for f in result_files])
        combined_result.to_csv(f"{args.output_directory}{os.sep}combined_results.csv", index=False,sep=";")

if __name__ == "__main__":
    main()