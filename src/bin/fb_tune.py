#!/usr/bin/env python3

import argparse
import logging
import os
import glob
import re
import json
import random

from typing import Union, Optional

from tqdm import tqdm

import torch
import numpy as np

from flat_bug.predictor import Predictor
from flat_bug.datasets import get_datasets
from flat_bug.coco_utils import fb_to_coco, split_annotations, filter_coco
from flat_bug.eval_utils import compare_groups
from flat_bug.config import write_cfg, read_cfg, DEFAULT_CFG

from scipy.optimize import differential_evolution
from skopt import gp_minimize

# Fixed ranges for the parameters during tuning - should probably be configurable
PARAMETER_RANGES = {
    "MIN_OBJ_SIZE": (0, 64),
    "MINIMUM_TILE_OVERLAP": (0, 512),
    "EDGE_CASE_MARGIN": (0, 192),
    "SCORE_THRESHOLD": (0.01, 0.5),
    "IOU_THRESHOLD": (0.01, 0.5)
}

# Class for scaling and unscaling the parameters - ensures that the parameters visible to the optimizer have equal dynamic ranges [0, 1]
class Scaler:
    def __init__(self, ranges):
        self.ranges = ranges
        self.scales = [(r[1] - r[0]) for r in ranges.values()]
        self.offsets = [r[0] for r in ranges.values()]

    def scale(self, params):
        """
        Scales the parameters
        """
        orig = params
        if not isinstance(params, list):
            params = params.tolist()
        value = [(p - o) / s for p, o, s in zip(params, self.offsets, self.scales)]
        return value
    
    def unscale(self, params):
        """
        Unscales the parameters
        """
        orig = params
        if not isinstance(params, list):
            params = params.tolist()
        value = [p * s + o for p, o, s in zip(params, self.offsets, self.scales)]
        return value

class Tuner(Predictor):
    def __init__(self, loader : torch.utils.data.DataLoader, default_cfg : dict, scale_before : Union[float, int], file_path : Optional[str], *args, **kwargs):
        self.loader = loader
        self.default_cfg = default_cfg
        self.scale_before = scale_before
        self.file_path = file_path
        if self.file_path is not None:
            self.file_path = os.path.abspath(self.file_path)
            # Check if path is a CSV
            if not self.file_path.endswith(".csv"):
                raise ValueError("The file path should end with '.csv'")
            # Check if the file exists, then remove it
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
        self.positional_args = (", ".join([str(a) for a in args]))
        if len(self.positional_args) > 0:
            self.positional_args += ", "
        self.kwargs = (", ".join([f"{k}={v}" for k, v in kwargs.items()]))
        if len(self.kwargs) > 0:
            self.kwargs = ", " + self.kwargs
        super().__init__(*args, **kwargs)
        self.set_hyperparameters(**self.default_cfg)

    def evaluate(self) -> float:
        costs = []
        weights = []
        for data in tqdm(self.loader, dynamic_ncols=True, leave=False, desc="Evaluating model "):
            image, labels = data
            prediction = list(split_annotations(fb_to_coco(self.pyramid_predictions(image, scale_before=self.scale_before).json_data, {}), True).values())[0]
            IoUs = compare_groups(
                group1              = labels, 
                group2              = prediction, 
                group_labels        = ["Ground Truth", "Predictions"],
                image_path          = None,
                output_identifier   = "TUNING",
                plot                = False,
                plot_scale          = 1,
                plot_boxes          = False,
                output_directory    = None,
                threshold           = self.IOU_THRESHOLD
            )["IoU"]
            if len(IoUs) == 0:
                costs.append(1)
            else:
                costs.append(1 - sum(IoUs) / len(IoUs))
            weights.append(len(IoUs))
        return sum([c * w for c, w in zip(costs, weights)]) / sum(weights)

    def cost(self, cfg : dict) -> float:
        self.set_hyperparameters(**cfg)
        cost = self.evaluate()
        if not hasattr(self, "cost_log"):
            self.cost_log = {col : [] for col in list(cfg.keys()) + ["COST"]}
        for k, v in cfg.items():
            self.cost_log[k].append(v)
        self.cost_log["COST"].append(cost)
        self.sync_data(1)
        self.set_hyperparameters(**self.default_cfg)
        return cost

    def sync_data(self, n):
        if n <= 0 or n > len(self.cost_log):
            raise ValueError("The argument n should be between 1 and the length of the data.")
        if not hasattr(self, "file_path") or self.file_path is None:
            return

        # Get the rows that need to be added
        new_data = {k : v[-n:] for k, v in self.cost_log.items()}

        # If the file doesn't exist, or if it is empty, write the header
        if not os.path.exists(self.file_path) or os.path.getsize(self.file_path) == 0:
            # Write new data including header if the file does not exist or is empty
            with open(self.file_path, "w") as f:
                f.write(";".join(new_data.keys()) + "\n")
        # Append the new data to the file
        with open(self.file_path, "a") as f:
            for i in range(n):
                f.write(";".join([str(v[i]) for v in new_data.values()]) + "\n")

    def __repr__(self) -> str:
        # Create a string representation of the tuner
        call = f"Tuner({self.positional_args}loader={self.loader}, default_cfg={self.default_cfg}, scale_before={self.scale_before}{self.kwargs})"
        if not hasattr(self, "cost_log"):
            return call
        # Get first and best cost and the associated parameter values
        first_cost = self.cost_log["COST"][0]
        best_cost = min(self.cost_log["COST"])
        best_idx = self.cost_log["COST"].index(best_cost)
        first_params = {k : v[0] for k, v in self.cost_log.items() if k != "COST"}
        best_params = {k : v[best_idx] for k, v in self.cost_log.items() if k != "COST"}
        # Create a string for the first and best results
        result_str = "{} Cost: {:.4f} achieved with parameters: {}"
        first_result = result_str.format("First", first_cost, first_params)
        best_result = result_str.format("Best", best_cost, best_params)
        # Create a summary string
        improvement = (first_cost - best_cost) / first_cost * 100
        summary = f"Optimization found minimum at {best_cost} with parameters {best_params}. Improvement over initial guess: {improvement:.1f}%."
        # Return the call, summary, first result and best result
        return "\n".join([call, summary, first_result, best_result])

class AnnotatedDataset(torch.utils.data.IterableDataset):
    FILES_PER_DATASET_PER_ITER = 1

    def __init__(self, files : list, annotations : dict, datasets_per_iter : Optional[int] = None, files_per_iter : Optional[int] = None):
        self.files = files
        self.datasets = get_datasets(files)
        self.file_dataset_idx = [i for f in files for i, v in enumerate(self.datasets.values()) if f in v]
        self.dataset_file_idx = {i : [] for i in range(len(self.datasets))}
        for i, v in enumerate(self.file_dataset_idx):
            self.dataset_file_idx[v].append(i)
        self.annotations = split_annotations(filter_coco(json.load(open(annotations, "r")), area=32**2), strip_directories=True)

        if datasets_per_iter is None:
            self.DATASETS_PER_ITER = len(self.datasets)
        else:
            self.DATASETS_PER_ITER = min(datasets_per_iter, len(self.datasets))
        if not files_per_iter is None:
            self.FILES_PER_DATASET_PER_ITER = files_per_iter

    def __getitem__(self, idx):
        image = self.files[idx]
        return image, self.annotations[os.path.basename(image)]
    
    def __len__(self):
        # return len(self.files)
        return self.DATASETS_PER_ITER * self.FILES_PER_DATASET_PER_ITER
    
    def __iter__(self):
        this_iter_dataset_idxs = random.sample(range(len(self.datasets)), k=self.DATASETS_PER_ITER)
        this_iter_idx = []
        [this_iter_idx.extend(random.sample(self.dataset_file_idx[i], k=min(self.FILES_PER_DATASET_PER_ITER, len(self.dataset_file_idx[i])))) for i in this_iter_dataset_idxs]

        for i in this_iter_idx:
            yield self[i]

def main():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="input_dir",
                            help="A directory that contains subdirectories for each COCO sub-datasets."
                                 "Each sub-dataset contains a single json file named 'instances_default.json' "
                                 "and the associated images", required=True)
    args_parse.add_argument("-a", "--annotations", dest="annotations", required=True)
    args_parse.add_argument("-p", "--input-pattern", dest="input_pattern", default=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
                            help=r"The pattern to match the images. Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$' i.e. jpg/jpeg/png case-insensitive.")
    args_parse.add_argument("-o", "--output-dir", dest="results_dir",
                            help="The result directory", required=True)
    args_parse.add_argument("-w", "--model-weights", dest="model_weights",
                            help="The .pt file", required=True)
    args_parse.add_argument("-s", "--scale-before", dest="scale_before", default=1.0, type=float,
                            help="Scale the image before inference."
                                  "Default is 1.0, i.e. no downscaling."
                                  "Changing this value will impact the results," 
                                  "and the tuned parameters should be used with the same value for inference.")
    args_parse.add_argument("-g", "--gpu", type=str, default="cuda:0", 
                            help="Which device to use for inference. Default is 'cuda:0', i.e. the first GPU.")
    args_parse.add_argument("-d", "--dtype", type=str, default="float16", 
                            help="Which dtype to use for inference. Default is 'float16'.")
    args_parse.add_argument("--init-cfg", type=str, default=None,
                            help="Path to a YAML containing the initial configuration guess or the string 'default' which will set the initial configuration guess to the default config."
                                 "Default is None, which is 1/3 of the range for each parameter.")
    args_parse.add_argument("-n", "--images-per-iter", type=int, default=1, 
                            help="Number of images per dataset used to estimate the cost. Default is 1.")
    args_parse.add_argument("--datasets-per-iter", type=int, default=None,
                            help="Number of datasets per iteration. Default is None, i.e. all datasets.")
    args_parse.add_argument("--max-iter", type=int, default=2,
                            help="Maximum number of iterations for the evolutionary optimization algorithm. Default is 2.")
    args_parse.add_argument("--pop-size", type=int, default=5,
                            help="Population size for the evolutionary optimization algorithm. Default is 5.")
    args_parse.add_argument("--method", type=str, default="bayesian",
                            help="Optimization algorithm to use. Default is 'bayesian'."
                                 "Options are 'evolutionary' or 'genetic' for differential evolution and 'bayesian'/'gaussian process'/'gp' for gaussian process optimization."
                                 "The maximum number of function evaluation for both methods is pop_size * (max_iter + 1) * number_of_parameters (5).")
    args_parse.add_argument("--mock", action="store_true",
                            help="Mock the tuning process. Doesn't load the model, but instead immediately returns the initial configuration guess.")
    args_parse.add_argument("--verbose", action="store_true",
                            help="Prints more information during the tuning process.")

    args = args_parse.parse_args()
    option_dict = vars(args)

    # Do we mock the tuning?
    mock = option_dict["mock"]
    
    # Get input options
    input_dir = option_dict["input_dir"]
    input_pattern = option_dict["input_pattern"]
    annotations = option_dict["annotations"]
    results_dir = option_dict["results_dir"]
    model_weights = option_dict["model_weights"]
    scale_before = option_dict["scale_before"]
    images_per_iter = option_dict["images_per_iter"]
    datasets_per_iter = option_dict["datasets_per_iter"]
    max_images = None # fixme: disabled for now

    # Get optimization options
    optimization_algorithm = option_dict["method"]
    init_cfg = option_dict["init_cfg"]
    max_iter = option_dict["max_iter"]
    pop_size = option_dict["pop_size"]
    max_fun = pop_size * (max_iter + 1) * sum([l != u for l, u in PARAMETER_RANGES.values()])

    # Get dtype and device
    device = torch.device(option_dict["gpu"])
    dtype = getattr(torch, option_dict["dtype"])

    # Parse the initial configuration guess
    if init_cfg == "default":
        init_cfg = DEFAULT_CFG
    elif isinstance(init_cfg, str):
        init_cfg = read_cfg(init_cfg)
    if isinstance(init_cfg, dict):
        init_cfg["MIN_OBJ_SIZE"] = init_cfg["MIN_MAX_OBJ_SIZE"][0]

    # Get the verbosity
    global verbose
    verbose = option_dict["verbose"]

    # Create a progress bar
    global pbar
    pbar = tqdm(
        total=max_fun,
        dynamic_ncols=True,
        leave=False,
        desc="Tuning",
        unit="evaluations"
    )

    # fixme: what to do here when mocking?
    if not mock:
        # Create the dataset for evaluating the tuning objective function
        files = sorted([f for f in glob.glob(os.path.join(input_dir, "**"), recursive=True) if re.search(input_pattern, f)])
        if max_images is not None:
            dataset_lens = [len(v) for v in get_datasets(files).values()]
            if max_images > sum(dataset_lens):
                logging.warning(f"max_images={max_images} is greater than the total number of images in the dataset. "
                                f"Setting max_images to {sum(dataset_lens)}")
            else:
                image_per_dataset = max_images // len(dataset_lens)
                total_images = sum([min(image_per_dataset, l) for l in dataset_lens])
                while total_images < max_images:
                    image_per_dataset += 1
                    total_images = sum([min(image_per_dataset, l) for l in dataset_lens])
                if total_images > max_images and verbose:
                    logging.warning(f"max_images={max_images} is not divisible* by the number of datasets. "
                                    f"Setting max_images to {total_images}")
                max_images = total_images
                files = [f for v in get_datasets(files).values() for f in v[:image_per_dataset]]
        dataset = AnnotatedDataset(files, annotations, datasets_per_iter=datasets_per_iter, files_per_iter=images_per_iter)
        # Get the model
        tuner = Tuner(loader=dataset, default_cfg=DEFAULT_CFG, scale_before=scale_before, file_path=str(os.path.join(results_dir, "tuning_log.csv")), model=model_weights, device=device, dtype=dtype)
    else:
        # If objective is called an error will be raised: "TypeError: 'NoneType' object is not callable"
        tuner = None

    # Conversion helper between list of number and configuration dictionary
    def create_cfg(params):
        if isinstance(params, (np.ndarray, torch.Tensor)):
            params = params.tolist()
        cfg = dict()
        for k, v in zip(PARAMETER_RANGES.keys(), params):
            if k in ["MIN_OBJ_SIZE", "MINIMUM_TILE_OVERLAP", "EDGE_CASE_MARGIN"]:
                v = int(v)
            if k == "MIN_OBJ_SIZE":
                cfg["MIN_MAX_OBJ_SIZE"] = (v, 10**8)
            else:
                cfg[k] = v
        return cfg
    
    # Initialize the parameter scaler - scales parameters to be 0 when at the lower bound and 1 when at the upper bound, and supplies a unscale function
    global scaler
    scaler = Scaler(PARAMETER_RANGES)

    # Define the objective function for the optimization algorithm    
    if not mock:
        def objective(params):
            global scaler, verbose, pbar
            pbar.update(1)
            cfg = create_cfg(scaler.unscale(params))
            if verbose:
                print(f"Trying configuration: {cfg}")
            cost = tuner.cost(cfg)
            if verbose:
                print(f"Cost={cost} for {cfg}")
            return cost
    else:
        # When mocking the tuning process, we just return a random number
        import random, time

        generator = random.Random(42)
        def mock_metric(x):
            return sum([abs(xi-1/2) for xi in x]) / len(x)

        def objective(x):
            global pbar
            pbar.update(1)
            time.sleep(0.01)
            true_cost = mock_metric(x)
            cost = true_cost + 2 * (generator.random() - 1/2) / 10 + 1/10
            if verbose:
                print(f"Cost={cost} for true cost={true_cost}")
            return cost
    
    # Define the arguments for the optimization algorithm
    if init_cfg is None:
        initial = [1/3 for _ in range(len(PARAMETER_RANGES))]
    else:
        initial = scaler.scale([init_cfg[k] for k in PARAMETER_RANGES.keys()])
        if any([i < 0 or i > 1 for i in initial]):
            raise ValueError("Initial configuration not within the defined parameter ranges:\n" + str(PARAMETER_RANGES))
    if verbose: 
        print(f"Initial configuration: {scaler.unscale(initial)}")
    lower_bound = scaler.scale([r[0] for r in PARAMETER_RANGES.values()])
    upper_bound = scaler.scale([r[1] for r in PARAMETER_RANGES.values()])
    bounds = [(l, u) for l, u in zip(lower_bound, upper_bound)]

    if optimization_algorithm in "evolutionary" or optimization_algorithm in "genetic":
        result = differential_evolution(objective, bounds, x0=initial, strategy="best1bin",
                                        maxiter=max_iter, popsize=pop_size,
                                        disp=True, polish=False, updating="immediate") 
    elif optimization_algorithm in "bayesian" or optimization_algorithm in "gaussian process" or optimization_algorithm in "gp":
        # Fixme: Initial guess seems to lead to worse convergence, perhaps this optimization algorithm isn't suited for providing an initial guess? For now we just ignore the initial guess
        result = gp_minimize(objective, bounds, x0=None, n_calls=max_fun, n_random_starts=min(10, max_fun), verbose=False)

    if mock:
        final_cost = mock_metric(result.x)
        init_cost = mock_metric(initial)
        improvement = (init_cost - final_cost) / init_cost * 100
        print(f"Mock optimization found minimum at {final_cost} with parameters {result.x}. True minimum should be [0.5, 0.5, 0.5, 0.5, 0.5]. Improvement over initial guess: {improvement:.1f}%.")
    else:
        print(tuner)

    pbar.close()

    if mock:
        class DUMMY:
            def __init__(self):
                self.x = np.array(initial)
        result = DUMMY()

    # Convert the best configuration to a dictionary and save it as a YAML
    result_values = create_cfg(scaler.unscale(result.x))
    if verbose:
        print(f"Best configuration: {result_values}")
    try:
        write_cfg(result_values, os.path.join(results_dir, "best_cfg.yaml"), overwrite=True)
    except Exception as e:
        if mock:
            print(f"Failed to save result config to {os.path.join(results_dir, 'best_cfg.yaml')} due to {str(e)}")
        else:
            raise e
        
if __name__ == "__main__":
    main()