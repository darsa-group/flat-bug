## OBS
# For this to work you have to:
#
#   1) Install `pyRemoteData` - `pip install pyremotedata`
#
#   2) Right now I forgot that ´h5py´ is a dependency, so you have to install that manually as well - `pip install h5py`
#
#   3) Install `lftp` - `sudo apt-get install lftp`
#
#   4) Configure SSH for ERDA and requires you to have a ERDA account WITHOUT a password for SFTP (this is fine as you have to use SSH keys anyway), you need to add the ssh-key to the ~/.ssh/config file. 
#      Details can be found at 'https://int.erda.au.dk/wsgi-bin/setup.py?topic=sftp' => 'Command-Line SFTP Access' => 'SFTP/LFTP on Windows, Mac OSX, Linux/UN*X'
#
#   5) Set up the `pyRemoteData` package (see https://github.com/asgersvenning/pyremotedata) and the correct environment variables PYREMOTEDATA_REMOTE_USERNAME etc.
#      PSA: Here it is probably easiest to understand in the beginning if PYREMOTEDATA_REMOTE_DIRECTORY is set to "/", i.e. in .bashrc 'export PYREMOTEDATA_REMOTE_DIRECTORY=/'. 
#      Otherwise the default IOHandler will start in the directory specified by PYREMOTEDATA_REMOTE_DIRECTORY, instead of the root directory.

import argparse
import logging
import os
from flat_bug.predictor import Predictor
import json
from pyremotedata.implicit_mount import *
from pyremotedata.dataloader import *
from tqdm import tqdm

if __name__ == '__main__':
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
    args_parse.add_argument("-s", "--scale-before", dest="scale_before", default=1.0, type=float,
                            help="Downscale the image before detection, but crops from the original image."
                                 "Note that the COCO dataset dimentions match the scaled image" # fixme, is that true?!
                            )
    args_parse.add_argument("-n", "--nmax", type=int, default=-1, help="Number of images to process\nSet to -1 to process all")

    args = args_parse.parse_args()
    option_dict = vars(args)

    # assert os.path.isdir(option_dict["input_dir"])
    assert os.path.isfile(option_dict["model_weights"])
    pred = Predictor(option_dict["model_weights"])

    # fixme, build from pred._model!
    categories = {"id": 1, "name": "insect"}

    coco_data = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [categories]  # Your category
    }

    with IOHandler(verbose = False, clean = False) as io:
        io.cd(option_dict["input_dir"])

        path_iterator = RemotePathIterator(
            io_handler = io,
            # These are basically network-performance parameters
            batch_size = 2, # How many files to download at once (larger is faster, but more memory intensive)
            batch_parallel = 1, # How many files are downloaded in parallel in during each batch (10 seems to be optimal for my connection, this is probably dependent on the amount of cores on the server)
            max_queued_batches = 10, # This relates to how much pre-fetching is done, i.e. how many batches are queued before the download is paused. This can be as large as you want, the larger the less stuttering you will have, but requires more local *disk* (NOT RAM) space
            n_local_files = 1 * 10 * 2, # This is parameter basically does the same as the one above, but it really needs to larger than batch_size * max_queued_batches, otherwise files will be deleted before they are used (This *will* result in an error). This parameter should probably be removed from the `pyRemoteData` package...
            clear_local = True, # Are local files temporary? I.e. should they be deleted after use?
            # These parameters are all related to file-indexing and filtering on the remote server
            override = False, # Should the file-index be re-generated? (has to be False if store is False - otherwise an error will be thrown)
            store = False, # This is important if we do not want to add files to the remote server (i.e. we only want to read them), if this is True, then the function will "cache" the file list in the directory in a file in the remote directory called "file_index.txt"
            pattern = r"^[^\/\.]+(\.jpg$|\.png$|\.jpeg$|\.JPG$|\.PNG$|\.JPEG)$", # TODO: Support more image formats, requires that the dataloader supports it! Currently as a hack, we skip files in subdirectories i.e. files with a '/' in their name, this is not ideal, as they are still read from the remote server
        )
        
        # Apply nmax, by truncating the list of remote paths if nmax != -1
        if option_dict["nmax"] != -1:
            assert option_dict["nmax"] > 0, ValueError(f"'--nmax'/'-n' must be positive not {option_dict['nmax']}") # It is not allowed to set nmax to any values less than 1, except -1 (which means all)
            path_iterator.remote_paths = path_iterator.remote_paths[:option_dict["nmax"]]  
        
        errors = 0
        for i, (local, remote) in tqdm(enumerate(path_iterator), desc="Processing images", total=len(path_iterator)):
            logging.info(f"Processing {os.path.basename(local)}")
            try:
                prediction = pred.pyramid_predictions(local, scale_before=option_dict["scale_before"])

                im_info, annots = prediction.coco_entry()
                im_info["id"] = i - errors
                for a in annots:
                    a["id"] = i - errors
                    a["image_id"] = i

                prediction.make_crops(out_dir=option_dict["results_dir"])
            except Exception as e:
                logging.error(f"Issue whilst processing {local}")
                #fixme, what is going on with /home/quentin/todo/toup/20221008_16-01-04-226084_raw_jpg.rf.0b8d397da3c47408694eeaab2cde06e5.jpg?
                logging.error(e)
                errors += 1
                raise e
            coco_data["images"].append(im_info)
            coco_data["annotations"].extend(annots)
        with open(os.path.join(option_dict["results_dir"], "coco_dataset.json"), "w") as json_file:
            json.dump(coco_data, json_file)

