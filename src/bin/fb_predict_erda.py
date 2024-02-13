#!/usr/bin/env python
## OBS
# For this to work you have to:
#
#   1) Install `pyRemoteData` - `pip install pyremotedata`
#
#   2) Right now I forgot that ´h5py´ is a dependency, so you have to install that manually as well - `pip install h5py`
#
#   3) Install `lftp` - `sudo apt-get install lftp`
#
#   4) Configure SSH for ERDA and requires you to have an ERDA account WITHOUT a password for SFTP (this is fine as you have to use SSH keys anyway), you need to add the ssh-key to the ~/.ssh/config file. 
#      Details can be found at 'https://int.erda.au.dk/wsgi-bin/setup.py?topic=sftp' => 'Command-Line SFTP Access' => 'SFTP/LFTP on Windows, Mac OSX, Linux/UN*X'
#
#   5) Set up the `pyRemoteData` package (see https://github.com/asgersvenning/pyremotedata) and the correct environment variables PYREMOTEDATA_REMOTE_USERNAME etc.
#      PSA: Here it is probably easiest to understand in the beginning if PYREMOTEDATA_REMOTE_DIRECTORY is set to "/", i.e. in .bashrc 'export PYREMOTEDATA_REMOTE_DIRECTORY=/'. 
#      Otherwise the default IOHandler will start in the directory specified by PYREMOTEDATA_REMOTE_DIRECTORY, instead of the root directory.

import argparse
import logging
import uuid

import os
import re

from tqdm import tqdm

import torch

from flat_bug.predictor import Predictor

from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *

if __name__ == '__main__':
    # Catch all instances of "ERROR:root:"exiftool" is not found, on path or as absolute path"
    logging.basicConfig(level=logging.CRITICAL)
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="input_dir",
                            help="A directory on ERDA that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images"
                            )
    args_parse.add_argument("-o", "--output-dir", dest="results_dir",
                            help="The result directory")
    args_parse.add_argument("-w", "--model-weights", dest="model_weights",
                            help="The .pt file")
    args_parse.add_argument("-s", "--scale-before", dest="scale_before", default=0.5, type=float,
                            help="Downscale the image before detection, but crops from the original image."
                                 "Default is 0.5, i.e. downscale the image to half the size before detection."
                            )
    args_parse.add_argument("-n", "--nmax", type=int, default=-1, help="Number of images to process\nSet to -1 to process all. ")
    args_parse.add_argument("-O", "--tile_overlap", type=int, default=384, help="Minimum overlap between tiles in pixels. Default is 384, lower is faster, but may miss larger instances.")
    args_parse.add_argument("-c", "--conf_threshold", type=float, default=0.2, help="Confidence threshold for predictions. Default is 0.2 (20%), the larger the faster, but more instances will be missed. A very low value can result in many false positives.")
    args_parse.add_argument("-f", "--fast", action="store_true", help="Use fast mode, which is faster, but may miss larger instances.")
    args_parse.add_argument("-g", "--gpu", type=str, default="cuda:0", help="Which device to use for inference. Default is 'cuda:0', i.e. the first GPU.")
    args_parse.add_argument("-d", "--dtype", type=str, default="float16", help="Which dtype to use for inference. Default is 'float16'.")
    args_parse.add_argument("-p", "--pattern", type=str, default=".*\\.jpg$", help="Which files to process. Default is '.*\\.jpg$'. Remember to use double escapes.")
    args_parse.add_argument("-N", "--name_pattern", type=str, default="{image_name}", help="How to name the output files. Default is '{image_name}', which will result in the output files being named after the image names.")
    args_parse.add_argument("-I", "--ignore_nesting_levels", type=int, default=0, help="How many levels of nesting to ignore. Default is 0, and has no effect. If set to 1, then the input directories {'a/b/c', 'a/b/d', 'a/b/e'} will be processed as the single directory 'a/b'.")


    args = args_parse.parse_args()
    option_dict = vars(args)
    fast, input_directory, output_directory, pattern, levels_ignored = option_dict["fast"], option_dict["input_dir"], option_dict["results_dir"], option_dict["pattern"], option_dict["ignore_nesting_levels"]
    if not os.path.isdir(output_directory):
        raise ValueError(f"Output directory '{output_directory}' does not exist.")
    if not os.path.isfile(option_dict["model_weights"]):
        raise ValueError(f"Model weights '{option_dict['model_weights']}' does not exist.")
    if levels_ignored < 0:
        raise ValueError(f"Levels ignored must be non-negative, not {levels_ignored}.")

    device = torch.device(option_dict["gpu"])
    dtype = getattr(torch, option_dict["dtype"])
    if not torch.cuda.is_available() and "cuda" in option_dict["gpu"]:
        raise ValueError(f"Device '{option_dict['gpu']}' is not available.")
    if dtype not in [torch.float16, torch.float32, torch.bfloat16]:
        raise ValueError(f"Dtype '{option_dict['dtype']}' is not supported.")
    
    pred = Predictor(option_dict["model_weights"], device=device, dtype=dtype)
    pred.MINIMUM_TILE_OVERLAP = option_dict["tile_overlap"]
    pred.SCORE_THRESHOLD = option_dict["conf_threshold"]

    with IOHandler(verbose = False, clean = False) as io:

        io.cd(input_directory)
        # Check for local file index
        local_file_index = os.path.join(os.getcwd(), output_directory, f"{option_dict['input_dir'].replace('/', '_')}_file_index.txt")
        if os.path.isfile(local_file_index):
            with open(local_file_index, "r") as file:
                file_index = [line.strip() for line in file.readlines()]
            io.cache["file_index"] = file_index

        path_iterator = RemotePathIterator(
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
            pattern = pattern # r"^[^\/\.]+(\.jpg$|\.png$|\.jpeg$|\.JPG$|\.PNG$|\.JPEG)$", # TODO: Currently as a hack, we skip files in subdirectories i.e. files with a '/' in their name, this is not ideal, as they are still read from the remote server
        )

        # Write local file index if it does not exist
        if not os.path.isfile(local_file_index):
            with open(local_file_index, "w") as file:
                file.write("\n".join(path_iterator.remote_paths))
        
        # Apply nmax, by truncating the list of remote paths if nmax != -1
        if option_dict["nmax"] != -1:
            assert option_dict["nmax"] > 0, ValueError(f"'--nmax'/'-n' must be positive not {option_dict['nmax']}") # It is not allowed to set nmax to any values less than 1, except -1 (which means all)
            path_iterator.remote_paths = path_iterator.remote_paths[:option_dict["nmax"]]

        dataset = RemotePathDataset(path_iterator, prefetch=16, device=device, dtype=dtype, return_local_path=True, gbif=False)
    
        errors = 0
        for i, (image, remote_path, local_path) in tqdm(enumerate(dataset), desc="Processing images", total=len(path_iterator), dynamic_ncols=True):
            image_name = os.path.basename(remote_path)
            # Retrieve the subdirectory of the image, and replace the separator with the local separator
            input_subdirectory = os.sep.join(re.sub(input_directory, "", remote_path).split("/")[:-1])
            # Remove possible trailing & leading "/"
            input_subdirectory = re.sub(r"^\/|\/$", "", input_subdirectory)
            input_directory_structure = input_subdirectory.split(os.sep)
            ####### THIS IS VERY SPECIFIC TO THE AMI DATASET STRUCTURE #######
            # Remove possible date directory
            if re.match(r"\d{8}|^\d{4}_\d{2}_\d{2}", input_directory_structure[-1]):
                input_directory_structure = input_directory_structure[:-1]
            trap_id = input_directory_structure[-1]
            ##################################################################
            image_base_name = os.path.splitext(image_name)[0]
            input_subdirectory = os.sep.join(input_directory_structure[:-levels_ignored]) if levels_ignored > 0 else input_subdirectory
            # Create a unique identifier for the image
            identifier = str(uuid.uuid4())
            # We create a subdirectory within the output directory for each subdirectory in the input directory, and then create subdirectories for every number of instances, within these subdirectories the results for each image is saved in separate folders with the name of the image. 
            # I.e. if we are running the model on the folder 'RRR/insects' with the output directory 'LLL/output' and the image 'RRR/insects/ants/ant_1.jpg' has 3 instances, then the result will be saved in 'LLL/output/insects/3/ant_1'.
            output_subdirectory = f'{output_directory}{os.sep}{input_subdirectory}'
            # Make sure the output directory exists
            if not os.path.isdir(output_subdirectory):
                os.makedirs(output_subdirectory)
            logging.info(f"Processing {remote_path} ({i+1}/{len(path_iterator)}) with identifier {identifier} and saving to {output_subdirectory}")
            try:
                # Run the model
                prediction = pred.pyramid_predictions(image, remote_path, scale_increment=1/2, scale_before=option_dict["scale_before"], single_scale=fast)
                # Save the results
                result_directory = prediction.save(
                    output_directory = output_subdirectory,
                    overview = os.path.join(output_subdirectory, "overviews"),
                    crops = os.path.join(output_subdirectory, "crops"),
                    metadata = os.path.join(output_subdirectory, "metadata"),
                    fast = fast,
                    mask_crops = False,
                    identifier = identifier,
                    basename = f'TRAPNAME_{trap_id}_IMAGENAME_{image_base_name}' # name_pattern.format(image_name=image_base_name, *reversed(input_directory_structure))
                )
            except Exception as e:
                logging.error(f"Issue whilst processing {remote_path}")
                #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
                logging.error(e)
                errors += 1
                raise e
        logging.info(f"Finished processing {len(path_iterator)} images from {input_directory} with {errors} errors.")

