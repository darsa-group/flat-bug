#!/usr/bin/env python3

import argparse
import logging
import os
import glob
import re
import json
import random

from typing import Union, Optional, Tuple, Dict, List

from tqdm import tqdm

import torch
import numpy as np

from flat_bug import logger
from flat_bug.predictor import Predictor
from flat_bug.datasets import get_datasets
from flat_bug.coco_utils import fb_to_coco, split_annotations, filter_coco
from flat_bug.eval_utils import compare_groups, best_confidence_threshold, f1_score
from flat_bug.config import write_cfg, read_cfg, DEFAULT_CFG

from scipy.optimize import differential_evolution
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

# Fixed ranges for the parameters during tuning - should probably be configurable
PARAMETER_RANGES = {
    "MIN_OBJ_SIZE": (1, 64),
    "IOU_THRESHOLD": (0.01, 0.9)
}

# Class for scaling and unscaling the parameters - ensures that the parameters visible to the optimizer have equal dynamic ranges [0, 1]
class Scaler:
    def __init__(self, ranges : Dict[str, Tuple[int, Union[int, float]]]):
        self.ranges = ranges
        self.scales = [(r[1] - r[0]) for r in ranges.values()]
        self.offsets = [r[0] for r in ranges.values()]

    def scale(self, params : Union[list, np.ndarray]) -> list:
        """
        Scales the parameters between 0 and 1. 

        scale(x) = (x - x_min) / (x_max - x_min)

        Args:
            params (Union[list, np.ndarray]): The parameters to scale

        Returns:
            list: The scaled parameters
        """
        if not isinstance(params, list):
            params = params.tolist()
        value = [(p - o) / s for p, o, s in zip(params, self.offsets, self.scales)]
        return value
    
    def unscale(self, params : Union[list, np.ndarray]) -> list:
        """
        Unscales values between 0 and 1 to the original parameter ranges.

        unscale(x) = x * (x_max - x_min) + x_min

        Args:
            params (Union[list, np.ndarray]): The scaled parameters, i.e. values between 0 and 1

        Returns:
            list: The unscaled parameters
        """
        if not isinstance(params, list):
            params = params.tolist()
        value = [p * s + o for p, s, o in zip(params, self.scales, self.offsets)]
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
        self.default_cfg.pop("SCORE_THRESHOLD", None)
        self._init_score_threshold = self.SCORE_THRESHOLD
    
    def evaluate(self) -> float:
        r"""
        Evaluates the model on the dataset(s) and returns the cost.
        Cost is defined as the average of one minus the intersection over union (IoU) for all matches between labels and predicted instances. This includes both matched predictions, unmatched predictions (false positives), and unmatched labels (false negatives).
        Including false positives and negatives ensures that the model is penalized for missing instances as well as for predicting instances that are not present in the ground truth.

        Mathematically the cost can be expressed as follows:

        :math:`C(L, P, M) = \frac{\mathlarger\sum\limits_{i=1}^N\;\mathlarger\sum\limits_{(p, q) \in M_i}\left(1 - \mathrm{IoU}\left(L_{i,p};\;P_{i,q}\right)\right)\cdot\mathbf{1}_{p\neq\emptyset\,\land\, q\neq\emptyset}}{\mathlarger\sum\limits_{i=1}^N \,\lvert M_i\rvert}`
        :math:`C: (L, P, M) \in \mathrm{List}(\mathrm{Poly}^*) \times \mathrm{List}(\mathrm{Poly}^*) \times \mathrm{List}(\mathcal{P}(\mathbb{Z}_+\times\mathbb{Z}_+)) \to [0, 1]`
        where:
        - :math:`L` is a list of the ground truth ***L***abels for each image
        - :math:`P` is a list of the ***P***redictions for each image
        - :math:`M` is a list of the set of ***M***atching pairs of indices between the ground truth and the predictions for each image
        - :math:`N` is the number of images, and the length of :math:`L`, :math:`P` and :math:`M`
        - :math:`IoU : (x, y) \rightarrow [0, 1]` is the intersection over union function between two instances (e.g. polygons or bounding boxes)
        False positives or negatives lead to either :math:`L_{i,p}` or :math:`P_{i,q}` being empty, which will result in a cost of 1 for that instance.
        A prediction perfectly matching a ground truth label will result in a cost of 0 for that instance.
        In reality, the cost is calculated as 1 minus the average IoU for each instance, this is equivalent to the above formula, but the code is a bit cleaner.
        Returns:
            float: The cost of the model on the dataset(s), where 0 corresponds to exactly finding and matching all ground truth instances, and 1 corresponds to not finding any instances.
        """
        eval_results = {}
        for data in tqdm(self.loader, dynamic_ncols=True, leave=False, desc="Evaluating model "):
            image, labels = data
            prediction = list(split_annotations(fb_to_coco(self.pyramid_predictions(image, scale_before=self.scale_before).json_data, {}), True).values())[0]
            this_results : dict = compare_groups(
                group1              = labels, 
                group2              = prediction, 
                group_labels        = ["Ground Truth", "Predictions"],
                image_path          = None,
                output_identifier   = "TUNING",
                plot                = False,
                plot_scale          = 1,
                plot_boxes          = False,
                output_directory    = None,
                threshold           = 1 / 3 # TODO: At what overlap with the label do we consider a prediction to be correct? 
            )
            # Merge the results
            for k, v in this_results.items():
                if k not in eval_results:
                    eval_results[k] = v
                else:
                    if isinstance(v, list):
                        eval_results[k].extend(v)
                    elif isinstance(v, np.ndarray):
                        eval_results[k] = np.concatenate([eval_results[k], v])
                    else:
                        raise ValueError(f"Unexpected type {type(v)} for {k}")
        # Update the cost threshold
        current_best_score_threshold = self.SCORE_THRESHOLD
        if len(eval_results["idx_1"]) > 25:
            current_best_score_threshold = max(0.01, best_confidence_threshold(eval_results["idx_1"] != -1, eval_results["IoU"], eval_results["conf2"]))
        # Calculate IoU Cost
        n, c_iou = 0, 0
        for idx_1, idx_2, iou, conf in zip(eval_results["idx_1"], eval_results["idx_2"], eval_results["IoU"], eval_results["conf2"]):
            if idx_2 != -1 and conf < current_best_score_threshold:
                # These cases are skipped, since they correspond to instances that would not have been predicted by the model with the new confidence threshold
                if idx_1 == -1:
                    continue # These are prior false positives, which are now correctly not predicted (true negatives)
                iou = 0 # These are prior true positives, which are now not predicted (false negatives)
            c_iou += 1 - iou
            n += 1
        c_iou = c_iou / n
        # Calculate F1 Cost
        labels = []
        predictions = []
        for i, (idx_1, idx_2, conf) in enumerate(zip(eval_results["idx_1"], eval_results["idx_2"], eval_results["conf2"])):
            if idx_2 != -1 and conf < current_best_score_threshold:
                if idx_1 == -1:
                    continue
                predictions.append(False)
                labels.append(True)
            else:
                predictions.append(idx_1 != -1)
                labels.append(idx_2 != -1)
        c_f1 = 1 - f1_score(np.asarray(labels), np.asarray(predictions))
        # Use the F1 cost
        cost = c_f1

        self.update_score_threshold(cost, current_best_score_threshold)
        return cost, c_f1, c_iou
    
    def update_score_threshold(self, cost, threshold):
        """
        Potentially updates the score threshold based on the cost and the current best score threshold.

        An update is applied if:
        - There are no prior costs
        - The current cost is the minimum and there are less than 5 prior costs
        - The current cost is below the mean of the costs below the median of the prior costs
        """
        prior_costs = [] if not hasattr(self, "cost_log") else self.cost_log["COST"]
        n_prior = len(prior_costs)
        do_update = (n_prior == 0) or (n_prior < 5 and cost < min(prior_costs)) or (n_prior > 5 and cost < np.mean([c for c in prior_costs if c < np.quantile(prior_costs, (n_prior ** 0.5) / n_prior)]))
        if do_update:
            self.set_hyperparameters(SCORE_THRESHOLD = (self.SCORE_THRESHOLD + threshold) / 2)


    def cost(self, cfg : dict) -> float:
        self.set_hyperparameters(**cfg)
        cost, c_f1, c_iou = self.evaluate()
        if not hasattr(self, "cost_log"):
            self.cost_log = {col : [] for col in list(cfg.keys()) + ["SCORE_THRESHOLD","COST", "COST_F1", "COST_IOU"]}
        for k, v in cfg.items():
            self.cost_log[k].append(v)
        self.cost_log["COST"].append(cost)
        self.cost_log["COST_F1"].append(c_f1)
        self.cost_log["COST_IOU"].append(c_iou)
        self.cost_log["SCORE_THRESHOLD"].append(self.SCORE_THRESHOLD)
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
        first_params["SCORE_THRESHOLD"] = self._init_score_threshold
        best_params = {k : v[best_idx] for k, v in self.cost_log.items() if k != "COST"}
        best_params["SCORE_THRESHOLD"] = self.SCORE_THRESHOLD
        # Create a string for the first and best results
        result_str = "{} Cost: {:.4f} achieved with parameters: {}"
        first_result = result_str.format("First", first_cost, first_params)
        best_result = result_str.format("Best", best_cost, best_params)
        # Create a summary string
        improvement = 0 if first_cost == 0 else (first_cost - best_cost) / first_cost * 100
        summary = f"Optimization found minimum at {best_cost} with parameters {best_params}. Improvement over initial guess: {improvement:.1f}%."
        # Return the call, summary, first result and best result
        return "\n".join(["", call, summary, first_result, best_result])

class AnnotatedDataset(torch.utils.data.IterableDataset):
    FILES_PER_DATASET_PER_ITER = 1

    def __init__(self, files : list, annotations : dict, datasets_per_iter : Optional[int] = None, files_per_iter : Optional[int] = None):
        self.files = []
        # Create a dictionary with the base name of the images as keys and the annotations as values
        self.annotations = split_annotations(filter_coco(json.load(open(annotations, "r")), area=32**2), strip_directories=True)
        # Add the files to the dataset if they are found in the annotations
        [self.files.append(file) if os.path.basename(file) in self.annotations else logging.warning(f"File {file} not found in the annotations!") for file in files]
        del files

        # Determine which files belong to which dataset
        self.datasets = get_datasets(self.files)
        self.file_dataset_idx = [i for f in self.files for i, v in enumerate(self.datasets.values()) if f in v]
        self.dataset_file_idx = {i : [] for i in range(len(self.datasets))}
        for i, v in enumerate(self.file_dataset_idx):
            self.dataset_file_idx[v].append(i)

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
        # Sample `DATASETS_PER_ITER` datasets
        this_iter_dataset_idxs = random.sample(range(len(self.datasets)), k=self.DATASETS_PER_ITER)
        # For each sampled dataset, sample `FILES_PER_DATASET_PER_ITER` files
        this_iter_idx = []
        [this_iter_idx.extend(random.sample(self.dataset_file_idx[i], k=min(self.FILES_PER_DATASET_PER_ITER, len(self.dataset_file_idx[i])))) for i in this_iter_dataset_idxs]

        # Yield the sampled files
        for i in this_iter_idx:
            yield self[i]

def main():
    args_parse = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    args_parse.add_argument("-i", "--input-data", dest="input_dir", required=True,
                            help="A directory that contains the images used for tuning.")
    args_parse.add_argument("-a", "--annotations", dest="annotations", required=True,
                            help="Path to a COCO-style JSON with annotations for the images in the input directory."
                            "The file will most likely have the base name 'instances_defaults.json'.")
    args_parse.add_argument("-p", "--input-pattern", dest="input_pattern", default=r"[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$",
                            help=r"The pattern to match the images. Default is '[^/]*\.([jJ][pP][eE]{0,1}[gG]|[pP][nN][gG])$' i.e. jpg/jpeg/png case-insensitive.")
    args_parse.add_argument("-o", "--output-dir", dest="results_dir", required=True,
                            help="The result directory to store the final hyperparameters and the tuning log.")
    args_parse.add_argument("-w", "--model-weights", dest="model_weights", required=True,
                            help="The path of the .pt file which contains the weights of the model to tune.")
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
    ## TODO: I think these 4 parameters are quite confusing and not user-friendly. 
    ## It would be much easier to understand one parameter for the accuracy of the cost estimate and one for the number of iterations to run the optimization algorithm.
    args_parse.add_argument("-n", "--images-per-iter", type=int, default=1, 
                            help="Number of images per dataset used to estimate the cost. Default is 1.")
    args_parse.add_argument("--datasets-per-iter", type=int, default=None,
                            help="Number of datasets per iteration. Default is None, i.e. all datasets.")
    args_parse.add_argument("--max-iter", type=int, default=2,
                            help="Maximum number of iterations for the evolutionary optimization algorithm. Default is 2.")
    args_parse.add_argument("--pop-size", type=int, default=5,
                            help="Population size for the evolutionary optimization algorithm. Default is 5.")
    ## END TODO
    args_parse.add_argument("--method", type=str, default="bayesian",
                            help="Optimization algorithm to use. Default is 'bayesian'."
                                 "Options are 'evolutionary' or 'genetic' for differential evolution and 'bayesian'/'gaussian process'/'gp' for gaussian process optimization."
                                 "The maximum number of function evaluation for both methods is pop_size * (max_iter + 1) * number_of_parameters (5).")
    args_parse.add_argument("--verbose", action="store_true",
                            help="Prints more information during the tuning process.")

    args = args_parse.parse_args()
    option_dict = vars(args)
    
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

    # Create the dataset for evaluating the tuning objective function
    files = sorted([f for f in glob.glob(os.path.join(input_dir, "**"), recursive=True) if re.search(input_pattern, os.path.basename(f))])
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
    def objective(params):
        global scaler, verbose, pbar
        pbar.update(1)
        cfg = create_cfg(scaler.unscale(params))
        if verbose:
            logger.info(f"Trying configuration: {cfg}")
        cost = tuner.cost(cfg)
        if verbose:
            logger.info(f"Cost={cost} for {cfg}")
        return cost
    
    # Define the arguments for the optimization algorithm - 
    if init_cfg is None:
        initial = [1/3 for _ in range(len(PARAMETER_RANGES))]
    else:
        initial = scaler.scale([init_cfg[k] for k in PARAMETER_RANGES.keys()])
        if any([i < 0 or i > 1 for i in initial]):
            raise ValueError("Initial configuration not within the defined parameter ranges:\n" + str(PARAMETER_RANGES))
    if verbose: 
        logger.info(f"Initial configuration: {scaler.unscale(initial)}")
    lower_bound = scaler.scale([r[0] for r in PARAMETER_RANGES.values()])
    upper_bound = scaler.scale([r[1] for r in PARAMETER_RANGES.values()])
    bounds = [(l, u) for l, u in zip(lower_bound, upper_bound)]

    # Create output directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)

    # Run the optimization algorithm
    if optimization_algorithm in "evolutionary" or optimization_algorithm in "genetic":
        result = differential_evolution(objective, bounds, x0=initial, strategy="best1bin",
                                        maxiter=max_iter, popsize=pop_size,
                                        disp=True, polish=False, updating="immediate") 
    elif optimization_algorithm in "bayesian" or optimization_algorithm in "gaussian process" or optimization_algorithm in "gp":
        result = gp_minimize(objective, bounds, x0=None, n_calls=max_fun, n_initial_points=min(10, max_fun), verbose=False)
        plot_convergence(result).get_figure().savefig(os.path.join(results_dir, "convergence_plot.png"))
        plot_objective(result).get_figure().savefig(os.path.join(results_dir, "objective_plot.png"))
    pbar.close()

    # Get the best configuration
    result_values = create_cfg(scaler.unscale(result.x))
    result_values["SCORE_THRESHOLD"] = tuner.SCORE_THRESHOLD
    tuner.set_hyperparameters(**result_values)

    # Print and save the results
    logger.info(tuner)
    write_cfg(result_values, os.path.join(results_dir, "best_cfg.yaml"), overwrite=True)
        
if __name__ == "__main__":
    main()