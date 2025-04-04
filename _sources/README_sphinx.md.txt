# <code> flatbug </code>
### **<center><ins>A General Method for Detection and Segmentation of Terrestrial Arthropods in Images</ins></center>**

<p align="center">
    <img src="_static/prediction.jpg" style="width: 75%;">
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darsa-group/flat-bug/blob/master/docs/flat-bug.ipynb)

`flatbug` is partly a high-performance pyramid tiling inference wrapper for [`YOLOv8`](https://github.com/ultralytics/ultralytics) and partly a hybrid instance segmentation dataset of terrestrial arthropods accompanied by an appropriate training schedule for `YOLOv8` segmentation models, built on top of the original [`YOLOv8` training schedule](https://docs.ultralytics.com/modes/train/#why-choose-ultralytics-yolo-for-training). 

The goal of `flatbug` is to provide a single unified model for detection and segmentation of all terrestrial arthropods on arbitrarily large images, especially fine-tuned for the case of top-down images/scans - thus the name `"flat"bug`.

### Installation
Installation via package managers coming later.
<!-- The latest version of `flatbug` can be installed with any of your favourite package managers such as:
#### `pip`
```py
python -m pip install flat-bug
```
#### `anaconda`
```py
conda install flat-bug -c conda-forge
```
#### `mamba`
```py
mamba install flat-bug -c conda-forge
```
#### `micromamba`
```py
micromamba install flat-bug -c conda-forge
``` -->
#### Source/development
Or a development version can be installed from source by cloning this repository:
```sh
git clone git@github.com:darsa-group/flat-bug.git
cd flat-bug
pip install -e .
```

However, as with other packages built with `PyTorch` it is best to ensure that `torch` is installed separately. See [https://pytorch.org/](https://pytorch.org/) for details. We recommend using `torch>=2.3`.

### CLI Usage
We provide a number of [CLI scripts](https://darsa.info/flat-bug/cli.html) with `flatbug`. The main one of interest is `fb_predict`, which can be used to run inference on images or videos:
```sh
fb_predict -i <DIR_WITH_IMGS> -o <OUTPUT_DIR> [-w <WEIGHT_PATH>] ...
```

### Tutorials
We provide a number of tutorials on general and advanced usage, training, deployment and hyperparameters of `flatbug` in [examples/tutorials](examples/tutorials) or with Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/darsa-group/flat-bug/blob/master/docs/flat-bug.ipynb).

### Documentation
Find our documentation at [https://darsa.info/flat-bug/](https://darsa.info/flat-bug/).

<!-- fixme: Remember to add this later! -->
<!-- ### Archive
#### Models

#### Data

### Contributions
#### Code

#### Data -->
