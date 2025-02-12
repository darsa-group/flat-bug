# Execute
To run the analyses simply open the "full_analysis.R" script and set the two
parameters as you need:

* `do_recompute`: Set to `FALSE`, unless you *really* want to recompute the
  aggregate statistics
* `dry_run`: Set to `TRUE` if you want to run the analyses

**OBS**: The analyses require internet access.

The first time you run the analyses will most likely take quite a while, 
depending on the internet connection of your machine, since we must download
all of the evaluation result files.

### RStudio
Open the script, and press "CTRL+SHIFT+S" use the "Source" command
(not to be confused with "Source with Echo") from the command palette 
(or top-right of the source/code pane).

### R Terminal
Run `source("full_analysis.R")`.

# Dependencies
There are number of dependencies necessary to run the analyses, most of which are
part of the tidyverse ecosystem. 

Finding R package dependencies ... Done!

### Install
```r
install.packages(
  c(
    "abind",
    "ape",
    "cli",
    "colorblindr",
    "colorspace",
    "coxed",
    "data.table",
    "extrafont",
    "forcats",
    "furrr",
    "future",
    "ggimage",
    "ggplot2",
    "ggpubr",
    "ggraph",
    "grid",
    "Hmisc",
    "magick",
    "magrittr",
    "memoise",
    "mgcv",
    "patchwork",
    "progressr",
    "quantreg",
    "RCurl",
    "scales",
    "stringr",
    "tidygraph",
    "tidyverse"
  )
)
```

### Dependency list
```txt
      abind | 1.4.8
        ape | 5.8
        cli | 3.6.3
colorblindr | 0.1.0
 colorspace | 2.1.1
      coxed | 0.3.3
 data.table | 1.15.4
  extrafont | 0.19
    forcats | 1.0.0
      furrr | 0.3.1
     future | 1.33.2
    ggimage | 0.3.3
    ggplot2 | 3.5.1
     ggpubr | 0.6.0
     ggraph | 2.2.1
       grid | 4.4.1
      Hmisc | 5.1.3
     magick | 2.8.3
   magrittr | 2.0.3
    memoise | 2.0.1
       mgcv | 1.9.1
  patchwork | 1.3.0
  progressr | 0.15.1
   quantreg | 5.98
      RCurl | 1.98.1.16
     scales | 1.3.0
    stringr | 1.5.1
  tidygraph | 1.3.1
  tidyverse | 2.0.0
```

### Recreate README dependency list
```r
library(magrittr)
library(dplyr)
library(furrr)
library(renv)
library(utils)

renv::dependencies() %>% 
    tibble::as_tibble() %>% 
    dplyr::distinct(Package) %>% 
    dplyr::arrange(tolower(Package)) %>% 
    dplyr::mutate(
        Version = purrr::map(Package, utils::packageVersion) %>% 
            purrr::map_chr(as.character),
        aligned_Package = stringr::str_c(
            strrep(" ", max(nchar(Package)) - nchar(Package)),
            Package
        ),
        fmt = stringr::str_glue("{aligned_Package} | {Version}")
    ) %>% 
    dplyr::summarize(
        install = stringr::str_c(
            "install.packages(\n  c(\n    ", 
            stringr::str_c('"', Package, '"', collapse=",\n    "), 
            "\n  )\n)"
        ),
        deps = stringr::str_c(
            "```txt\n", 
            stringr::str_c(fmt, collapse="\n"), 
            "\n```"
        )
    ) %>% 
    dplyr::mutate(
        out = stringr::str_glue("\n\n### Install\n```r\n{install}\n```\n\n### Dependency list\n{deps}")
    ) %>% 
    pull(out) %>% 
    cat
```