import math
import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.pyplot import Axes
from matplotlib.ticker import FormatStrFormatter
from torchvision.io import ImageReadMode, read_image
from tqdm import tqdm as TQDM

from flat_bug import download_from_repository
from flat_bug.datasets import get_datasets
from flat_bug.predictor import Predictor


### Timing functions
def file_to_tensor(path : str, device : torch.types.Device):
    return read_image(
        path=path, 
        mode=ImageReadMode.RGB, 
        apply_exif_orientation=True
    ).to(device)

def time_model(size : str, files : list[str], device="cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.float16) -> dict[str, tuple[float, int, tuple[int, int]]]:
    torch.cuda.empty_cache()

    model = Predictor(f"flat_bug_{size}.pt", device=device, dtype=dtype)
    with TemporaryDirectory() as tmpdir:
        tmpfiles = {lfile : download_from_repository(file, lfile) for file in files if (lfile := os.path.join(tmpdir, os.path.basename(file)))}
        timings = dict()
        for path in TQDM(tmpfiles.keys(), desc=f"Predicting ({size})..."):
            torch.cuda.empty_cache()
            image = file_to_tensor(path, device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            instances = len(model(image))
            end.record()
            torch.cuda.synchronize(device)
            timings[os.path.basename(path)] = (start.elapsed_time(end) / 1000, instances, tuple(image.shape[1:])) 
            del image
    totals = list(map(sum, list(zip(*timings.values()))[:2]))
    print('Found {} instances in {} images in {:.1f} seconds'.format(totals[1], len(tmpfiles), totals[0]))
    
    del model, tmpfiles
    torch.cuda.empty_cache()
    
    return timings

### Plotting functions
# --- A simple lowess implementation using only NumPy ---
def lowess(x, y, frac=0.3):
    """
    A simple lowess smoother using a tricube weighting kernel.
    x and y must be 1D arrays of the same length.
    frac is the fraction of points used for local regression.
    Returns an array of smoothed y-values.
    """
    n = len(x)
    r = int(np.ceil(frac * n))
    y_smoothed = np.zeros(n)
    for i in range(n):
        # Compute distances from x[i]
        distances = np.abs(x - x[i])
        # Bandwidth: the r-th smallest distance
        h = np.sort(distances)[r]
        # Avoid division by zero; if h is zero, just use the point itself.
        if h == 0:
            y_smoothed[i] = y[i]
            continue
        # Compute tricube weights
        w = (1 - (distances / h) ** 3) ** 3
        w[distances > h] = 0
        # Compute weighted linear regression
        sum_w = np.sum(w)
        if sum_w == 0:
            y_smoothed[i] = y[i]
        else:
            # Weighted means
            xw = np.sum(w * x) / sum_w
            yw = np.sum(w * y) / sum_w
            # Weighted slope
            b_num = np.sum(w * (x - xw) * (y - yw))
            b_den = np.sum(w * (x - xw) ** 2)
            b = b_num / b_den if b_den != 0 else 0
            a = yw - b * xw
            y_smoothed[i] = a + b * x[i]
    return y_smoothed

# --- Plotting function for one model ---
def plot_results(results: dict[str, tuple[float, int, tuple[int, int]]],
                 ax: Axes,
                 color: str = 'C0',
                 label: str = None,
                 scale=math.sqrt,
                 scale_inv=lambda x: x ** 2,
                 min_area=5,
                 max_area=300):
    """
    Plots the scatter points for one model.
    
    results: dict mapping keys to (seconds, instances, (height, width))
    ax: matplotlib Axes object
    color: marker color
    label: label for the model (used in the color legend)
    scale, scale_inv: functions for scaling the instance count (e.g., sqrt and square)
    min_area, max_area: marker area limits (points^2)
    
    Returns:
        x: array of x-values (sqrt(pixel area))
        y: array of y-values (seconds)
        instances: list of instance counts
    """
    # Unpack data: seconds, instance count, (height, width)
    seconds, instances, dimensions = list(zip(*results.values()))
    # x-values: square root of pixel area
    areas = [h * w for h, w in dimensions]
    x = np.array([a ** 0.5 for a in areas])
    y = np.array(seconds) * 1000 / instances
    
    # Scale instance counts
    scaled_instances = list(map(scale, instances))
    scaled_min, scaled_max = min(scaled_instances), max(scaled_instances)
    if scaled_max == scaled_min:
        norm = [1 for _ in scaled_instances]
    else:
        norm = [(sv - scaled_min) / (scaled_max - scaled_min) for sv in scaled_instances]
    sizes = [min_area + nv * (max_area - min_area) for nv in norm]
    
    # Plot the points
    ax.scatter(x, y, s=sizes, facecolors=color, alpha=0.7, edgecolors='none',
               label=label)
    
    return x, y, instances

if __name__ == "__main__":
    # Preparation
    with NamedTemporaryFile("r") as f:
        download_from_repository("folder_index.txt", f.name)
        files = f.readlines()

    files = [f.strip() for f in files if "fb_yolo/insects/images/val" in f]
    ds = get_datasets(files)
    files_per_ds = 5
    files = []
    for fs in ds.values():
        this_files = sorted(fs)[:min(files_per_ds, len(fs))]
        if isinstance(this_files, list):
            files.extend(this_files)
        else:
            files.append(this_files)

    ## Timing
    model_results = {s : time_model(s, files) for s in "LMSN"}

    ## Plotting
    # Define model color mapping
    model_colors = {"L": "C0", "M": "C1", "S": "C2", "N": "C3"}

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # For a shared marker size legend, we collect all instance counts.
    all_instances = []
    data_by_model = {}  # to store each model's (x, y, instances)
    for model_key in model_colors:
        x, y, inst = plot_results(
            model_results[model_key],
            ax=ax,
            color=model_colors[model_key],
            label=model_key,  # used for color legend
            scale=math.sqrt,
            scale_inv=lambda x: x ** 2
        )
        data_by_model[model_key] = (x, y, inst)
        all_instances.extend(inst)

    # Add lowess (trend) lines for each model.
    for model_key, (x, y, inst) in data_by_model.items():
        # Sort by x for a smooth line
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]
        y_smooth = lowess(x_sorted, y_sorted, frac=0.25)
        ax.plot(x_sorted, y_smooth, color=model_colors[model_key],
                linestyle='--', linewidth=2, label=f"{model_key} trend")

    ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.set_xlabel("Square root of image pixel area")
    ax.set_ylabel("Milliseconds per instance")

    # Format the y-axis to display plain numbers.
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # --- Build combined legends ---

    # 1. Color legend for the models.
    # Create dummy handles (Line2D markers) for color mapping.
    color_handles = []
    for model_key, col in model_colors.items():
        handle = plt.scatter([], [],
                            color=col, s=100,
                            edgecolors="none",
                            label=model_key)
        color_handles.append(handle)
    # Place this legend centered vertically near the top half.
    legend_color = ax.legend(handles=color_handles, title="Model\nsize",
                            loc="center left", bbox_to_anchor=(1.02, 0.75),
                            borderaxespad=0.5)

    # 2. Size legend for instance counts.
    # Global scaling based on all instance counts (using sqrt scaling).
    all_scaled = list(map(math.sqrt, all_instances))
    global_scaled_min, global_scaled_max = min(all_scaled), max(all_scaled)

    def global_instance_to_size(i, min_area=5, max_area=300):
        scaled_i = math.sqrt(i)
        if global_scaled_max == global_scaled_min:
            v = 1
        else:
            v = (scaled_i - global_scaled_min) / (global_scaled_max - global_scaled_min)
        return min_area + v * (max_area - min_area)

    def global_size_to_instance(s, min_area=5, max_area=300):
        norm_val = (s - min_area) / (max_area - min_area)
        # Invert the sqrt scaling by squaring
        instance_val = (norm_val * (global_scaled_max - global_scaled_min) + global_scaled_min) ** 2
        return int(round(instance_val))

    # Choose instance values spanning the global range.
    min_inst_leg = 10 ** math.floor(math.log10(min(all_instances)))
    max_inst_leg = 10 ** math.ceil(math.log10(max(all_instances)))
    # Compute corresponding marker sizes.
    legend_sizes = np.linspace(global_instance_to_size(min_inst_leg),
                            global_instance_to_size(max_inst_leg), 5)
    legend_instance_labels = [f"{global_size_to_instance(s)}" for s in legend_sizes]
    size_handles = [plt.scatter([], [], s=s, color="gray", alpha=0.7, edgecolors='none')
                    for s in legend_sizes]
    # Place this legend centered vertically near the bottom half.
    legend_size = ax.legend(handles=size_handles, labels=legend_instance_labels,
                            title="Instances", loc="center left", bbox_to_anchor=(1.02, 0.25),
                            labelspacing=1.25,
                            borderaxespad=0.5, borderpad=1)

    # Add the color legend back so both are shown.
    ax.add_artist(legend_color)

    plt.tight_layout()
    plt.savefig("time_flatbug_figure.png", transparent=True)

    def time_per_instance(results):
        ms = [1000 * t/n for t, n, _ in results.values()]
        return sum(ms) / len(ms), float(np.median(ms)), min(ms), max(ms)

    time_summary = {
        s : time_per_instance(model_results[s])
        for s in "LMSN"
    }

    lines = ["model,mean_ms_per_instance,median_ms_per_instance,min_ms_per_instance,max_ms_per_instance\n"]
    for s, (tavg, tmed, tmin, tmax) in time_summary.items():
        lines.append(f'{s},{tavg},{tmed},{tmin},{tmax}\n')
    print("Timing results:\n")
    print(*lines)
    with open("model_timing.csv", "w") as f:
        f.writelines(lines)