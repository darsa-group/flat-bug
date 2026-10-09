# flat-bug

**Detection and segmentation of terrestrial arthropods in images.**

flat-bug finds every arthropod in an image and outlines it, whatever the image: a sticky
card, a pitfall-trap tray, a camera-trap frame, a museum drawer, a 500-megapixel scan. One
model covers them all. It is tuned for animals seen from above on a more or less flat surface,
hence the name.

```{image} _static/prediction.jpg
:alt: flat-bug predictions drawn on a sticky card: every insect outlined.
:class: fb-hero
```

flat-bug is two things:

- **A predictor** that runs a segmentation model over a pyramid of overlapping tiles, so it
  sees animals of a few pixels and of thousands of pixels in the same image, at any image size,
  and merges what it finds into one set of outlines.
- **A model zoo and the dataset behind it**: a compilation of annotated datasets from traps,
  scanners, cameras and collections, and the models trained on them. See [](models.md).

```{raw} html
<div class="fb-cards">
  <a class="fb-card" href="user_guide.html"><b>User guide</b><span>Install, predict, read the results, train.</span></a>
  <a class="fb-card" href="models.html"><b>Models</b><span>Every released version, how it was trained, how it scores.</span></a>
  <a class="fb-card" href="api.html"><b>API reference</b><span>The Python package and command-line tools.</span></a>
</div>
```

## Quick start

```bash
uv pip install flat-bug --torch-backend=auto
fb_predict -i my_images/ -o results/
```

Each image gets a folder with the outlines as JSON, an overview image and a crop of every
animal. The [user guide](user_guide.md) has the details.

## Cite flat-bug

If flat-bug is useful in your work, please cite the paper:

> Svenning, A., Mougeot, G., Alison, J., Chevalier, D., Chavez Molina, N., Ong, S.-Q., Bjerge, K.,
> Carrillo, J., Høye, T. T., & Geissmann, Q. (2026). A general method for detection and segmentation
> of terrestrial arthropods in images. *Methods in Ecology and Evolution*, 17(3), 727–739.
> [doi:10.1111/2041-210X.70249](https://doi.org/10.1111/2041-210x.70249)

```bibtex
@article{svenning2026flatbug,
  title   = {A general method for detection and segmentation of terrestrial arthropods in images},
  author  = {Svenning, Asger and Mougeot, Guillaume and Alison, Jamie and Chevalier, Daphne and
             Chavez Molina, Nisa and Ong, Song-Quan and Bjerge, Kim and Carrillo, Juli and
             H{\o}ye, Toke Thomas and Geissmann, Quentin},
  journal = {Methods in Ecology and Evolution},
  volume  = {17},
  number  = {3},
  pages   = {727--739},
  year    = {2026},
  doi     = {10.1111/2041-210X.70249}
}
```

The training data is published as a single collection, with a DOI for each dataset in it:
[doi:10.5281/zenodo.14761446](https://doi.org/10.5281/zenodo.14761446). If you use one of
those datasets, cite it too.

## Authors

flat-bug is developed in the [DARSA group](https://darsa.info) at Aarhus University, Denmark.
The paper's authors are:

- Asger Svenning ([ORCID](https://orcid.org/0009-0007-4615-9972)), Department of Ecoscience, Aarhus University
- Guillaume Mougeot, Department of Ecoscience, Aarhus University
- Jamie Alison, Department of Ecoscience, Aarhus University
- Daphne Chevalier, Centre for Sustainable Food Systems, University of British Columbia
- Nisa Chavez Molina, Centre for Sustainable Food Systems, University of British Columbia
- Song-Quan Ong, Department of Ecoscience, Aarhus University
- Kim Bjerge ([ORCID](https://orcid.org/0000-0001-6742-9504)), Department of Electrical and Computer Engineering, Aarhus University
- Juli Carrillo ([ORCID](https://orcid.org/0000-0003-4673-2875)), Centre for Sustainable Food Systems, University of British Columbia
- Toke Thomas Høye ([ORCID](https://orcid.org/0000-0001-5387-3284)), Department of Ecoscience, Aarhus University
- Quentin Geissmann ([website](https://quentin.geissmann.net)), Center for Quantitative Genetics and Genomics, Aarhus University

## Contribute data

flat-bug gets better with every new kind of image it is trained on. If you have annotated
images of arthropods, or images flat-bug struggles with, send them through the
[data contribution form](https://forms.gle/hQe2dzLs4tHcCarEA). Code, issues and discussion are
on [GitHub](https://github.com/darsa-group/flat-bug).

```{toctree}
:hidden:

user_guide
models
api
```
