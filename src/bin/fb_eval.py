
import os
from glob import glob
import json
from tqdm import tqdm
from flat_bug.coco_utils import fb_to_coco, split_annotations
from flat_bug.eval_utils import compare_groups
import argparse

if __name__ == "__main__":

    # # Development defaults
    # predictions = "dev/**/**.json"
    # ground_truth = "s3/CollembolAI/instances_default.json"
    # image_directory = "s3/CollembolAI"
    # output_directory = "dev/eval"
    # iou_match_threshold = 0.1

    parser = argparse.ArgumentParser(description='Evaluate predictions')
    parser.add_argument('-p', '--predictions', type=str, help='Path or pattern to the predictions files')
    parser.add_argument('-g', '--ground_truth', type=str, help='Path to the ground truth file')
    parser.add_argument('-I', '--image_directory', type=str, help='Path to the image directory')
    parser.add_argument('-o', '--output_directory', type=str, help='Path to the output directory')
    parser.add_argument('-M', '--iou_match_threshold', type=float, default=0.1, help='IoU match threshold. Defaults to 0.1')
    parser.add_argument('-P', '--plot', action="store_true", help='Plot the matches and the IoU matrix')
    parser.add_argument('-b', '--no_boxes', action="store_false", help='Do not plot the bounding boxes')
    parser.add_argument('-c', '--coco_predictions', action="store_true", help='Whether the predictions are already in a COCO format (legacy)')
    parser.add_argument('-s', '--scale', type=float, default=1, help='Scale of the output images. Defaults to 1. Lower is faster.')
    parser.add_argument('-n', type=int, default=-1, help='Number of images to process. Defaults to -1 (all images)')

    args = parser.parse_args()

    if args.coco_predictions:
        pred_coco = json.load(open(args.predictions, "r"))
    else:
        files = sorted(glob(args.predictions, recursive=True))
        flat_bug_predictions = [json.load(open(p)) for p in files]
        pred_coco = {}
        for d in flat_bug_predictions:
            fb_to_coco(d, pred_coco)

    if not os.path.exists(args.ground_truth):
        raise ValueError(f'Ground truth file not found: {args.ground_truth}')
    gt_coco = json.load(open(args.ground_truth, "r"))
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
        print(f'Ground truth has {len(gt_diff_keys)} images that are not in the predictions: [{missing_gt_diff_formatted}{", ..." if len(gt_diff_keys) > show else ""}] and {len(gt_diff_keys) - show} more')
    if len(pred_diff_keys) > 0:
        show = min(2, len(pred_diff_keys))
        missing_pred_diff_formatted = ', '.join(['"' + str(i) + '"' for i in pred_diff_keys[:show]])
        print(f'Predictions has {len(pred_diff_keys)} images that are not in the ground truth: [{missing_pred_diff_formatted} {", ..." if len(pred_diff_keys) > show else ""}] and {len(pred_diff_keys) - show} more')
    if len(shared_keys) == 0:
        raise ValueError(f'No images in common between the ground truth and the predictions')

    shared_keys = sorted(shared_keys)
    if args.n != -1:
        print(f'Skipping the evaluation of {len(shared_keys) - args.n} images')
        shared_keys = shared_keys[:args.n]

    for image in tqdm(shared_keys, desc="Evaluating images", dynamic_ncols=True):
        matches = compare_groups(
            group1              = gt_annotations[image], 
            group2              = pred_annotations[image], 
            group_labels        = ["Ground Truth", "Predictions"],
            image_path          = f'{args.image_directory}{os.sep}{image}', 
            output_identifier   = os.path.splitext(image)[0], 
            plot                = args.plot,
            plot_scale          = args.scale,
            plot_boxes          = args.no_boxes,
            output_directory    = args.output_directory,
            threshold           = args.iou_match_threshold
        )