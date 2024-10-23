#!/usr/bin/env python3

import argparse
import os
import glob
import re

from typing import Optional, List

from tqdm import tqdm

from flat_bug import logger, set_log_level
from flat_bug.coco_utils import fb_to_coco
from flat_bug.predictor import Predictor
from flat_bug.config import read_cfg, DEFAULT_CFG

import torch


def cli_args():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", type=str, dest="input_dir", required=True,
                            help="A directory that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images"
                            )
    args_parse.add_argument("-o", "--output-dir", type=str, dest="output_dir", required=True,
                        help="The result directory")
    args_parse.add_argument("-w", "--model-weights", type=str, dest="model_weights", required=True,
                            help="The .pt file")
    args_parse.add_argument("-p", "--input-pattern", type=str, dest="input_pattern", default=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
                            help=r"The pattern to match the images. Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$' i.e. jpg/jpeg/png case-insensitive.")
    args_parse.add_argument("-n", "--max-images", type=int, dest="max_images", default=None,
                            help="Maximum number of images to process. Default is None. Truncates in alphabetical order.")
    args_parse.add_argument("-s", "--scale-before", type=float, dest="scale_before", default=1.0,
                            help="Downscale the image before detection, but crops from the original image."
                                 "Note that the COCO dataset dimentions match the scaled image" # fixme, is that true?!
                            )
    args_parse.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Which device to use for inference. Default is 'cuda:0', i.e. the first GPU.")
    args_parse.add_argument("-d", "--dtype", type=str, default="float16", help="Which dtype to use for inference. Default is 'float16'.")
    args_parse.add_argument("-f", "--fast", action="store_true", help="Use fast mode.")
    args_parse.add_argument("--config", type=str, default=None, help="The config file.")
    args_parse.add_argument("--no-crops", action="store_true", help="Do not save the crops.")
    args_parse.add_argument("--no-overviews", action="store_true", help="Do not save the overviews.")
    args_parse.add_argument("--no-metadata", action="store_true", help="Do not save the metadata.")
    args_parse.add_argument("--only-overviews", action="store_true", help="Only save the overviews.")
    args_parse.add_argument("--long-format", action="store_true", help="Use long format for storing results.")
    args_parse.add_argument("-S", "--no-save", action="store_true", help="Do not save the results.")
    args_parse.add_argument("-C", "--no-compiled-coco", action="store_true", help="Skip the production of a compiled COCO file (for all images).")
    args_parse.add_argument("--single-scale", action="store_true", help="Use single scale.")
    args_parse.add_argument("--verbose", action="store_true", help="Verbose mode.")
    
    args = args_parse.parse_args()
    return vars(args)

def main(
        input_dir : str,
        output_dir : str,
        model_weights : str,
        input_pattern : str=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
        max_images : Optional[int]=None,
        scale_before : float=1.0,
        gpu : str="cuda:0",
        dtype : str="float16",
        fast : bool=False,
        config : Optional[str]=None,
        no_crops : bool=False,
        no_overviews : bool=False,
        no_metadata : bool=False,
        only_overviews : bool=False,
        long_format : bool=False,
        no_save : bool=False,
        no_compiled_coco : bool=False,
        single_scale : bool=False,
        verbose : bool=False
    ):
    logger.debug("OPTIONS:", locals())

    isERDA = input_dir.startswith("erda://")
    if isERDA:
        from pyremotedata.implicit_mount import IOHandler, RemotePathIterator
        logger.debug("Assuming directory exists on ERDA")
    else:
        _, ext = os.path.splitext(input_dir)
        isVideo = ext in [".mp4", ".avi"]
        if not isVideo:
            if not os.path.isdir(input_dir):
                raise FileNotFoundError(f"Directory '{input_dir}' not found.")
    assert os.path.isfile(model_weights)

    device = gpu
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
    
    dtype = dtype
    
    config = config
    if config:
        config = read_cfg(config)
    else:
        config = DEFAULT_CFG
    
    crops = not no_crops
    metadata = not no_metadata
    if no_overviews:
        if only_overviews:
            raise ValueError("Cannot set both --no-overviews and --only-overviews.")
        overviews = False
    elif only_overviews:
        if long_format:
            raise ValueError("Cannot set both --only-overviews and --long-format. --only-overviews already saves in long format (although not the same file structure as --long-format).")
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

    verbose = verbose
    if verbose:
        config["TIME"] = True
        set_log_level("DEBUG")

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
        input_dir = input_dir.removeprefix("erda://")
        io = IOHandler(verbose = False, clean = False)
        io.start()
        io.cd(input_dir )
        # Check for local file index
        local_file_index = os.path.join(os.getcwd(), output_dir, f"{input_dir.replace('/', '_')}_file_index.txt")
        if os.path.isfile(local_file_index):
            with open(local_file_index, "r") as file:
                file_index = [line.strip() for line in file.readlines()]
            io.cache["file_index"] = file_index

        file_iter = RemotePathIterator(
            io_handler = io,
            # These are basically network-performance parameters
            batch_size = 64, # How many files to download at once (larger is faster, but more memory intensive)
            batch_parallel = 10, # How many files are downloaded in parallel in during each batch (10 seems to be optimal for my connection, this is probably dependent on the amount of cores on the server)
            max_queued_batches = 3, # This relates to how much pre-fetching is done, i.e. how many batches are queued before the download is paused. This can be as large as you want, the larger the less stuttering you will have, but requires more local *disk* (NOT RAM) space
            n_local_files = 100 * 3 * 2, # This is parameter basically does the same as the one above, but it really needs to larger than batch_size * max_queued_batches, otherwise files will be deleted before they are used (This *will* result in an error). This parameter should probably be removed from the `pyRemoteData` package...
            clear_local = False, # Are local files temporary? I.e. should they be deleted after use? TODO: This should also cause the previous argument to be ignored, and **never** delete files before internally
            # These parameters are all related to file-indexing and filtering on the remote server
            override = False, # Should the file-index be re-generated? (has to be False if store is False - otherwise an error will be thrown)
            store = False, # This is important if we do not want to add files to the remote server (i.e. we only want to read them), if this is True, then the function will "cache" the file list in the directory in a file in the remote directory called "file_index.txt"
            pattern = input_pattern # r"^[^\/\.]+(\.jpg$|\.png$|\.jpeg$|\.JPG$|\.PNG$|\.JPEG)$", # TODO: Currently as a hack, we skip files in subdirectories i.e. files with a '/' in their name, this is not ideal, as they are still read from the remote server
        )
    elif isVideo:
        import cv2
        import tempfile
        tmp_frame_dir = tempfile.TemporaryDirectory()
        video_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(input_dir))[0] + ".mp4")
        cap = cv2.VideoCapture(input_dir)
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
        file_iter = sorted([f for f in glob.glob(os.path.join(input_dir, "**"), recursive=True) if re.search(input_pattern, f)])
    if max_images is not None:
        if isERDA:
            file_iter.subset(list(range(min(max_images, len(file_iter)))))
        else:
            file_iter = file_iter[:max_images]

    all_json_results = []

    UUID = "ChangeThisTEMPORARY" # fixme, this is a temporary solution, but we should use a UUID for each run
    
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
            prediction = pred.pyramid_predictions(f, scale_increment=1/2, scale_before=scale_before, single_scale=single_scale)
            # Save the results
            if not no_save:
                result_directory = prediction.save(
                    output_directory = output_dir,
                    fast = fast,
                    overview = overviews,
                    metadata = metadata,
                    crops = crops,
                    mask_crops = True,
                    identifier = UUID, #str(uuid.uuid4()),
                )
                if not result_directory is None:
                    json_files = [f for f in  glob.glob(os.path.join(result_directory, "*.json"))]
                    assert len(json_files) == 1
                    all_json_results.append(json_files[0])
                if isVideo and overviews:
                    if not result_directory is None:
                        overview_file = glob.glob(os.path.join(result_directory, f"overview_*UUID_{UUID}.jpg"))
                        if len(overview_file) == 1:
                            overview_file = overview_file[0]
                        elif len(overview_file) > 1:
                            raise ValueError("Multiple overview files found.")
                        elif len(overview_file) == 0:
                            raise ValueError("No overview file found.")
                        else:
                            raise ValueError(f"Unexpected error. Found {len(overview_file)} overview files?")
                    elif isinstance(overviews, str):
                        overview_file = os.path.join(overviews, f"overview_{os.path.splitext(os.path.basename(f))[0]}_UUID_{UUID}.jpg")
                    else:
                        raise ValueError(f"Unexpected video inference settings. {result_directory=}, {overviews=}")
                    frames.append(overview_file)
        except Exception as e:
            logger.error(f"Issue whilst processing {f}")
            #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
            logger.error(e)
            raise e
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

if __name__ == "__main__":
    main(**cli_args())