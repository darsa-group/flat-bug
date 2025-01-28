# Execute
To run the analyses simply open the "full_analysis.R" script and set the two
parameters as you need:

* `do_recompute`: Set to `FALSE`, unless you *really* want to recompute the
  aggregate statistics
* `dry_run`: Set to `TRUE` if you want to run the analyses

**OBS**: The analyses require internet access.

### RStudio
Open the script, and press "CTRL+SHIFT+S" use the "Source" command
(not to be confused with "Source with Echo") from the command palette 
(or top-right of the source/code pane).

### R Terminal
Run `source("full_analysis.R")`.

# Dependencies
There are number of dependencies necessary to run the analyses, most of which are
part of the tidyverse ecosystem. 

### Install
```r
install.packages(
  c(
    "forcats",
    "grid",
    "Hmisc",
    "magick",
    "scales",
    "extrafont",
    "ggpubr",
    "mgcv",
    "RCurl",
    "tidyverse",
    "data.table",
    "furrr",
    "cli",
    "progressr",
    "coxed",
    "googlesheets4",
    "memoise",
    "quantreg",
    "stringr",
    "magrittr",
    "patchwork",
    "colorblindr",
    "colorspace",
    "ggplot2",
    "abind",
    "ape",
    "ggimage",
    "ggraph",
    "tidygraph",
    "future"
  )
)
```

### Dependency list
```txt
forcats [1.0.0]
grid [4.4.1]
Hmisc [5.1.3]
magick [2.8.3]
scales [1.3.0]
extrafont [0.19]
ggpubr [0.6.0]
mgcv [1.9.1]
RCurl [1.98.1.16]
tidyverse [2.0.0]
data.table [1.15.4]
furrr [0.3.1]
cli [3.6.3]
progressr [0.15.1]
coxed [0.3.3]
googlesheets4 [1.1.1]
memoise [2.0.1]
quantreg [5.98]
stringr [1.5.1]
magrittr [2.0.3]
patchwork [1.3.0]
colorblindr [0.1.0]
colorspace [2.1.1]
ggplot2 [3.5.1]
abind [1.4.8]
ape [5.8]
ggimage [0.3.3]
ggraph [2.2.1]
tidygraph [1.3.1]
future [1.33.2]
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
    dplyr::mutate(
        Version = purrr::map(Package, utils::packageVersion) %>% 
            purrr::map_chr(as.character),
        fmt = stringr::str_glue("{Package} [{Version}]")
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
        out = stringr::str_glue("\n\n## Install\n```r\n{install}\n```\n\n## Dependency list\n{deps}")
    ) %>% 
    pull(out) %>% 
    cat
```