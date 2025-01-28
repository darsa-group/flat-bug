library(furrr)
library(RCurl)

# Compare backbone sizes / Deployment models
plan(multisession, workers = round(availableCores() / 2))

tibble(
  size = c("L", "M", "S", "N") %>% 
    factor(., .),
  local_old = paste0("data/compare_backbone_sizes_", size, ".csv"),
  local_full = paste0("data/compare_backbone_sizes_", size, "_full.csv"),
  remote_old = train_repository %>% 
    paste0("/", size) %>% 
    paste0("_ucloud_done_V2/compare_backbone_sizes/eval/combined_results.csv"),
  remote_full = train_repository %>% 
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
      curlPerform(url = path, writedata = local@ref, noprogress = T)
      close(local)
      return(invisible())
    }, .progress = T),
    result_full = future_map2(local_full, remote_full, function(local, path) {
      if (file.exists(local)) return(invisible())
      local <- CFILE(local, "wb")
      curlPerform(url = path, writedata = local@ref, noprogress = T)
      close(local)
      return(invisible())
    }, .progress = T)
  )

plan(sequential)

# Leave-one-out
paste0(
  train_repository,
  "/leave_one_out_cv_and_finetuning_done/leave_one_out_cv_and_finetuning/eval/combined_results.csv"
) %>% 
  download.file(
    destfile = "data/leave_one_out_combined.csv", 
    quiet = T
  )

plan(multisession, workers = round(availableCores() / 2))

"data/leave_one_out_combined.csv" %>% 
  read_csv(show_col_types = F) %>%
  pull(model) %>%
  unique %>%
  str_split("/") %>%
  map_chr(~ .x[5]) %>% 
  future_map(function(name) {
    f <- paste0("data/", name, ".csv")
    if (file.exists(f)) return(invisible())
    f <- CFILE(f, "wb")
    r <- paste0(
      train_repository,
      "/leave_one_out_cv_and_finetuning_done/leave_one_out_cv_and_finetuning/eval/", 
      name,
      "/eval/combined_results.csv"
    )
    curlPerform(url = r, writedata = f@ref, noprogress = T)
    close(f)
    return(invisible())
  }, .progress = T) %>% 
  invisible()

plan(sequential)

# Leave-two-out
paste0(
  train_repository,
  "/leave_two_out_dataset_mapping_done/leave_two_out_dataset_mapping/eval/combined_results.csv"
) %>%
  download.file(
    destfile = "data/leave_two_out_combined.csv",
    quiet = T
  )

plan(multisession, workers = round(availableCores() / 2))

"data/leave_two_out_combined.csv" %>% 
  read_csv(show_col_types = F) %>%
  pull(model) %>%
  unique %>%
  str_split("/") %>%
  map_chr(~ .x[5]) %>% 
  future_map(function(name) {
    f <- paste0("data/", name, ".csv")
    if (file.exists(f)) return(invisible())
    f <- CFILE(f, "wb")
    r <- paste0(
      train_repository,
      "/leave_two_out_dataset_mapping_done/leave_two_out_dataset_mapping/eval/", 
      name,
      "/eval/combined_results.csv"
    )
    curlPerform(url = r, writedata = f@ref, noprogress = T)
    close(f)
    return(invisible())
  }, .progress = T) %>% 
  invisible()

plan(sequential)

# Precomputed recomputed statistics
precomputed_files <- c(
  "leave_two_out_combined_recomputed_clean.csv",
  "leave_two_out_combined_recomputed_clean.rds",
  "leave_two_out_combined_recomputed.csv",
  "leave_one_out_combined_recomputed.csv",
  "compare_backbone_sizes_combined_recomputed.csv"
)

list(
  str_c(fb_repository, "manuscript", "data", precomputed_files, sep = "/"),
  file.path("data", precomputed_files)
) %>% 
  pmap(function(src, dst, ...) {
    if (file.exists(dst)) return(invisible())
    f <- CFILE(dst, "wb")
    r <- src
    curlPerform(url = r, writedata = f@ref, noprogress = T)
    close(f)
    return(invisible())
  }, .progress = "progressr")
