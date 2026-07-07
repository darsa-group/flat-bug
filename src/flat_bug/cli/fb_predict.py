#!/usr/bin/env python3
r"""Inference CLI script for ``flatbug``.

A comprehensive CLI API for ``flatbug`` inference with support for hyperparameter configuration, 
flexible input parsing, output format specification, and hardware specification.

Usage:
    ``fb_predict -i INPUT_PATH_OR_DIRECTORY -o OUTPUT_DIRECTORY [OPTIONS]``

Options:
    -h, --help            show this help message and exit
    -i INPUT, --input INPUT
                        A image file or a directory of image files
    -o OUTPUT_DIR, --output OUTPUT_DIR
                        The result directory
    -w MODEL_WEIGHTS, --model-weights MODEL_WEIGHTS
                        The .pt file
    -p INPUT_PATTERN, --input-pattern INPUT_PATTERN
                        The pattern to match the images. 
                        Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$' i.e. jpg/jpeg/png case-insensitive.
    -n MAX_IMAGES, --max-images MAX_IMAGES
                        Maximum number of images to process. Default is None. Truncates in alphabetical order.
    -R, --recursive       Process images nested within subdirectories of the input.
    -s SCALE_BEFORE, --scale-before SCALE_BEFORE
                        Downscale the image before detection, but crops from the original image.
    --single-scale        Use single scale.
    -g GPU, --gpu GPU     Which device to use for inference. Default is 'cuda:0', i.e. the first GPU.
    -d DTYPE, --dtype DTYPE
                        Which dtype to use for inference. Default is 'float16'.
    -f, --fast            Use fast mode.
    --config CONFIG       The config file.
    --no-crops            Do not save the crops.
    --no-overviews        Do not save the overviews.
    --no-metadata         Do not save the metadata.
    --only-overviews      Only save the overviews.
    --long-format         Use long format for storing results.
    -S, --no-save         Do not save the results.
    -C, --no-compiled-coco
                        Skip the production of a compiled COCO file (for all images).
    -v, --verbose         Verbose mode.
"""

import argparse
import glob
import os
import re
import uuid

import torch
from tqdm import tqdm

from flat_bug import logger, set_log_level
from flat_bug.coco_utils import fb_to_coco
from flat_bug.config import DEFAULT_CFG, read_cfg
from flat_bug.predictor import Predictor
from flat_bug.predictor import _executor as prediction_executor

# TODO: fixme
# ruff: disable[D103]


def cli_args():
    args_parse = argparse.ArgumentParser(
        prog="fb_predict",
        description="""\
            Perform instance detection and segmentation with flatbug on
            one or more images or a video.""",
        formatter_class=argparse.RawTextHelpFormatter
    )

    args_parse.add_argument(
        "-i", "--input", type=str, dest="input", required=True,
        help="A image file or a directory of image files"
    )
    args_parse.add_argument(
        "-o", "--output", type=str, dest="output_dir", required=True,
        help="The result directory"
    )
    args_parse.add_argument(
        "-w", "--model-weights", type=str, dest="model_weights", default="flat_bug_M_v2.pt",
        help="The .pt file"
    )
    args_parse.add_argument(
        "-p", "--input-pattern", type=str, dest="input_pattern", default=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
        help=(
            "The pattern to match the images. "
            r"Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$' i.e. jpg/jpeg/png case-insensitive."
    ))
    args_parse.add_argument(
        "-n", "--max-images", type=int, dest="max_images", default=None,
        help="Maximum number of images to process. Default is None. Truncates in alphabetical order."
    )
    args_parse.add_argument(
        "-R", "--recursive", action="store_true", 
        help="Process images nested within subdirectories of the input."
    )
    args_parse.add_argument(
        "-s", "--scale-before", type=float, dest="scale_before", default=1.0,
        help="Downscale the image before detection, but crops from the original image."
    )
    args_parse.add_argument(
        "--single-scale", action="store_true", help="Use single scale."
    )
    args_parse.add_argument(
        "-M", "--nms_metric", type=str, default=None,
        help=(
            "Overlap metric to use for NMS, if specified this will override the config. "
            "Default is 'IoU', currently only 'IoS' is also available."
    ))
    args_parse.add_argument(
        "-g", "--device", "--gpu", type=str, default="auto", 
        help="Which device to use for inference."
    )
    args_parse.add_argument(
        "-d", "--dtype", type=str, default=None,
        help="Which dtype to use for inference. Default is 'float16' for CUDA and 'float32' for CPU."
    )
    args_parse.add_argument(
        "-f", "--fast", action="store_true", help="Use fast mode."
    )
    args_parse.add_argument(
        "--config", type=str, default=None, help="The config file."
    )
    args_parse.add_argument(
        "--id", type=str, default=None, required=False, help="Identifier (ID) for prediction run."
    )
    args_parse.add_argument(
        "--no-crops", action="store_true", help="Do not save the crops."
    )
    args_parse.add_argument(
        "--no-overviews", action="store_true", help="Do not save the overviews."
    )
    args_parse.add_argument(
        "--no-metadata", action="store_true", help="Do not save the metadata."
    )
    args_parse.add_argument(
        "--only-overviews", action="store_true", help="Only save the overviews."
    )
    args_parse.add_argument(
        "--long-format", action="store_true", help="Use long format for storing results."
    )
    args_parse.add_argument(
        "-S", "--no-save", action="store_true", help="Do not save the results."
    )
    args_parse.add_argument(
        "-C", "--no-compiled-coco", action="store_true",
        help="Skip the production of a compiled COCO file (for all images)."
    )
    args_parse.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose mode."
    )
    
    args = args_parse.parse_args()
    return vars(args)

def predict(
        input : str,
        output_dir : str,
        model_weights : str,
        input_pattern : str=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
        max_images : int | None=None,
        recursive : bool=False,
        scale_before : float=1.0,
        single_scale : bool=False,
        nms_metric : str="IoU",
        device : str="auto",
        dtype : str=None,
        fast : bool=False,
        config : str | None=None,
        id : str | None=None,
        no_crops : bool=False,
        no_overviews : bool=False,
        no_metadata : bool=False,
        only_overviews : bool=False,
        long_format : bool=False,
        no_save : bool=False,
        no_compiled_coco : bool=False,
        verbose : bool=False
    ):
    if verbose:
        set_log_level("DEBUG")
    
    torch.set_float32_matmul_precision("medium")
    
    logger.debug(f"OPTIONS: {locals()}")

    # Sanitize paths
    isVideo = False
    isERDA = input.startswith("erda://")
    if not isERDA: 
        input = os.path.normpath(input)
    output_dir = os.path.normpath(output_dir)
    model_weights = os.path.normpath(model_weights)
    if config is not None:
        config = os.path.normpath(config)

    if isERDA:
        from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
        logger.debug("Assuming directory exists on ERDA")
    else:
        _, ext = os.path.splitext(input)
        isVideo = ext in [".mp4", ".avi"]
        if not isVideo:
            if not os.path.exists(input):
                raise FileNotFoundError(f"Directory '{input}' not found.")

    if device is None or device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info("CUDA available, using GPU")
        else:
            device = "cpu"
            logger.info("CUDA not available, using CPU")
    
    if not torch.cuda.is_available() and "cuda" in device:
        raise ValueError(f"Device(s) '{device}' is/are not available.")
    # Detect if multi-gpu, either by comma or semicolon
    if "," in device:
        device = device.split(",")
    elif ";" in device:
        device = device.split(";")
    if isinstance(device, list):
        device = [f"cuda:{d}" if d.isdigit() else d for d in device]
        device = [torch.ones(1).to(torch.device(d)).device for d in device]
    else:
        device = f"cuda:{device}" if device.isdigit() else device
        device = torch.ones(1).to(torch.device(device)).device
    device_type = set([d.type for d in (device if isinstance(device, list) else [device])])
    if len(device_type) != 1:
        raise RuntimeError("Unable to resolve device type.")
    device_type = list(device_type)[0].lower().strip()
    if device_type not in ["cpu", "cuda"]:
        logger.warning(f"Unsupported device type: {device_type} - unexpected behavior or crashes may arise.")
    if dtype is None:
        if device_type == "cpu":
            dtype = "float32"
        else:
            dtype = "float16"
    dtype = dtype
    
    if config is not None:
        config = read_cfg(config)
    else:
        config = DEFAULT_CFG
    if verbose:
        config["TIME"] = device_type == "cuda"
    if nms_metric is not None:
        config["OVERLAP_METRIC"] = nms_metric
    
    if id is None:
        id = str(uuid.uuid4())
    
    crops = not no_crops
    metadata = not no_metadata
    if no_overviews:
        if only_overviews:
            raise ValueError("Cannot set both --no-overviews and --only-overviews.")
        overviews = False
    elif only_overviews:
        if long_format:
            raise ValueError(
                "Cannot set both --only-overviews and --long-format. "
                "--only-overviews already saves in long format "
                "(although not the same file structure as --long-format)."
            )
        overviews = output_dir
        crops = False
        metadata = False
    else:
        overviews = True

    if long_format:
        if overviews:
            overviews = os.path.join(output_dir, "overviews")
        if crops:
            crops = os.path.join(output_dir, "crops")
        if metadata:
            metadata = os.path.join(output_dir, "metadata")

    pred = Predictor(model_weights, device=device, dtype=dtype, cfg=config)

    # fixme, build from pred._model!
    categories = {"id": 1, "name": "insect"}

    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [categories]  # Your category
    }
    if isERDA:
        input = input.removeprefix("erda://")
        io = IOHandler(verbose = False, clean = False)
        io.start()
        io.cd(input)
        # Check for local file index
        local_file_index = os.path.join(os.getcwd(), output_dir, f"{input.replace(os.sep, '_')}_file_index.txt")
        if os.path.isfile(local_file_index):
            with open(local_file_index) as file:
                file_index = [line.strip() for line in file.readlines()]
            io.cache["file_index"] = file_index

        file_iter = RemotePathIterator(
            io_handler = io,
            # These are basically network-performance parameters
            # How many files to download at once (larger is faster, but more memory intensive)
            batch_size = 64,
            # How many files are downloaded in parallel in during each batch (10 seems to be optimal for my connection, 
            # this is probably dependent on the amount of cores on the server)
            batch_parallel = 10,
            # This relates to how much pre-fetching is done, i.e. how many batches are queued before the download is paused.
            # This can be as large as you want, the larger the less stuttering you will have, but requires more local *disk* (NOT RAM) space
            max_queued_batches = 3,
            # This is parameter basically does the same as the one above, 
            # but it really needs to larger than batch_size * max_queued_batches, 
            # otherwise files will be deleted before they are used (This *will* result in an error). 
            # This parameter should probably be removed from the `pyRemoteData` package...
            n_local_files = 100 * 3 * 2,
            # Are local files temporary? I.e. should they be deleted after use?
            # TODO: This should also cause the previous argument to be ignored, and **never** delete files before internally
            clear_local = False,
            # These parameters are all related to file-indexing and filtering on the remote server
            # Should the file-index be re-generated? (has to be False if store is False - otherwise an error will be thrown)
            override = False,
            # This is important if we do not want to add files to the remote server (i.e. we only want to read them), 
            # if this is True, then the function will "cache" the file list in 
            # the directory in a file in the remote directory called ".file_index.txt"
            store = False,
            # r"^[^\/\.]+(\.jpg$|\.png$|\.jpeg$|\.JPG$|\.PNG$|\.JPEG)$",
            # # TODO: Currently as a hack, we skip files in subdirectories 
            # i.e. files with a '/' in their name, this is not ideal, as they are still read from the remote server
            pattern = input_pattern
        )
    elif isVideo:
        import tempfile

        import cv2
        tmp_frame_dir = tempfile.TemporaryDirectory()
        video_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input))[0] + ".mp4")
        cap = cv2.VideoCapture(input)
        fps = cap.get(cv2.CAP_PROP_FPS) 
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        video_shape = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        file_iter = []
        # Make progress bar that shows the progress in time
        pbar = tqdm(total=duration, desc="Reading video frames", dynamic_ncols=True, unit="s")
        while cap.isOpened():
            pbar.update(1/fps)
            ret, frame = cap.read()
            if not ret:
                break
            # Write frame as file in tmp_frame_dir
            tmp_file = os.path.join(tmp_frame_dir.name, f"{len(file_iter)}.jpg")
            cv2.imwrite(tmp_file, frame)
            file_iter.append(tmp_file)
        cap.release()
        pbar.close()
        frames = []
        if not no_save and overviews:
            # Create a video writer
            if fast:
                video_shape = (video_shape[0]//2, video_shape[1]//2)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, video_shape)
    else:
        if os.path.isfile(input):
            file_iter = [input]
        else:
            file_iter = sorted([f for f in glob.glob(os.path.join(input, "**"), recursive=recursive) if re.search(input_pattern, f)])
    if max_images is not None:
        if isERDA:
            file_iter.subset(list(range(min(max_images, len(file_iter)))))
        else:
            file_iter = file_iter[:max_images]

    all_json_results = []
    
    pbar = tqdm(enumerate(file_iter), total=len(file_iter), desc="Processing images", dynamic_ncols=True, unit="image")
    for i, f in pbar:
        if isERDA:
            tmp_file, file_name = f
            f = tmp_file
            # f = os.path.join(output_dir, file_name)
            # os.rename(tmp_file, f)
        if verbose:
            logger.info(f"Processing {os.path.basename(f)}")
        pbar.set_postfix_str(f"Processing {os.path.basename(f)}")
        try:
            # Run the model
            prediction = pred.pyramid_predictions(f, scale_increment=2/3, scale_before=scale_before, single_scale=single_scale)
            # Save the results
            if not no_save:
                result_directory = prediction.save(
                    output_directory = output_dir,
                    fast = fast,
                    overview = overviews,
                    metadata = metadata,
                    crops = crops,
                    mask_crops = True,
                    identifier = id,
                )
                if result_directory is not None:
                    basename = os.path.splitext(os.path.basename(f))[0]
                    metadata_directory = metadata if isinstance(metadata, str) else result_directory
                    overview_directory = overviews if isinstance(overviews, str) else result_directory
                    # crop_directory = crops if isinstance(crops, str) else os.path.join(result_directory, crops)
                    all_json_results.append(os.path.join(metadata_directory, f'metadata_{basename}_UUID_{id}.json'))
                    if isVideo and overviews:
                        frames.append(os.path.join(overview_directory, f"overview_{basename}_UUID_{id}.jpg"))
        except Exception:
            #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
            logger.exception(f"Issue whilst processing {f}")
            raise
    if verbose:
        logger.info("Finalizing results...")
    prediction_executor.flush(progress=True)
    if verbose:
        logger.info("All results finished.")
    if not no_compiled_coco:
        if len(all_json_results) == 0:
            logger.info("No results found, unable to compile COCO file.")
        else:
            import json

            compiled_coco = os.path.join(output_dir, "coco_instances.json")
            pred_coco = {}
            
            flat_bug_predictions = [json.load(open(p)) for p in all_json_results]
            for d in flat_bug_predictions:
                fb_to_coco(d, pred_coco)
            with open(compiled_coco,"w") as f:
                json.dump(pred_coco, f)
    if isVideo and frames and not no_save and overviews:
        for frame in tqdm(frames, desc=f"Writing video ({video_output_path})", unit="frame"):
            img = cv2.imread(frame)
            if fast:
                img = cv2.resize(img, (video_shape[0], video_shape[1]))
            video_writer.write(img)
        video_writer.release()
        tmp_frame_dir.cleanup()
    if pred._multi_gpu:
        raise NotImplementedError("Multi-GPU support is not supported. Worker termination is not implemented.")
    if isERDA:
        io.stop()
    if verbose:
        logger.info("All steps done, process cleaning up.")

def main():
    kwargs = cli_args()

    if kwargs.get('gpu', None) is not None:
        logger.warning("'gpu' argument is deprecated!")
        if kwargs.get("device", None) not in [None, "auto"]:
            raise RuntimeError("Supplying both 'gpu' and 'device' is ambigous. Please use only one, preferably 'device'.")
        kwargs["device"] = kwargs.pop("gpu")
    else:
        kwargs.pop("gpu", None)

    predict(**kwargs)

if __name__ == "__main__":
    main()