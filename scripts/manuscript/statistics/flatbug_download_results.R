source("flatbug_init.R")
library(furrr)
library(RCurl)

# Compare backbone sizes / Deployment models

plan(multisession, workers = round(availableCores() / 2))

tibble(
  size = c("L", "M", "S", "N") %>%
    factor(., .),
  local_old = paste0("data/compare_backbone_sizes_", size, ".csv"),
  local_full = paste0("data/compare_backbone_sizes_", size, "_full.csv"),
  remote_old = erda_repository %>%
    paste0("/", size) %>%
    paste0("_ucloud_done_V2/compare_backbone_sizes/eval/combined_results.csv"),
  remote_full = erda_repository %>%
    paste0("/", size) %>%
    paste0(
      "_ucloud_done_V2/compare_backbone_sizes/eval/fb_compare_backbone_sizes_",
      size,
      "/eval/combined_results.csv"
    )
) %>%
  mutate(
    result_old = future_map2(local_old, remote_old, function(local, path) {
      if (file.exists(local)) return(invisible())
      local <- CFILE(local, "wb")
      curlPerform(url = path, writedata = local@ref, noprogress = TRUE)
      close(local)
      return(invisible())
    }, .progress = TRUE),
    result_full = future_map2(local_full, remote_full, function(local, path) {
      if (file.exists(local)) return(invisible())
      local <- CFILE(local, "wb")
      curlPerform(url = path, writedata = local@ref, noprogress = TRUE)
      close(local)
      return(invisible())
    }, .progress = TRUE)
  )

plan(sequential)

# Leave-one-out

paste0(
  erda_repository,
  "/leave_one_out_cv_and_finetuning_done/leave_one_out_cv_and_finetuning/eval/combined_results.csv"
) %>%
  download.file(destfile = "data/leave_one_out_combined.csv")

plan(multisession, workers = round(availableCores() / 2))

read_csv("data/leave_one_out_combined.csv") %>%
  pull(model) %>%
  unique %>%
  str_split("/") %>%
  map_chr(~ .x[5]) %>%
  future_map(function(name) {
    f <- paste0("data/", name, ".csv")
    if (file.exists(f)) return(invisible())
    f <- CFILE(f, "wb")
    r <- paste0(
      erda_repository,
      "/leave_one_out_cv_and_finetuning_done/leave_one_out_cv_and_finetuning/eval/",
      name,
      "/eval/combined_results.csv"
    )
    curlPerform(url = r, writedata = f@ref, noprogress = TRUE)
    close(f)
    return(invisible())
  }, .progress = TRUE) %>%
  invisible()

plan(sequential)
