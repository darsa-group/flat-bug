library(furrr)

recompute_stats <- function(file, size_threshold, boot.n=1000) {
  full_data <- data.table::fread(file) %>% 
    as_tibble %>% 
    mutate(
      area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
      size = sqrt(area),
      dataset = str_extract(image, "^[^_]+"),
      short = short_name(dataset),
      result = case_when(
        idx_1 != -1 & idx_2 == -1 ~ "FN",
        idx_1 == -1 & idx_2 != -1 ~ "FP",
        idx_1 != -1 & idx_2 != -1 ~ "TP"
      )
    ) %>% 
    filter(size >= size_threshold) %>% 
    select(dataset, short, result)
  
  calc_stats_ <- function(data) {
    data %>% 
      group_by(dataset, short) %>%
      slice_sample(
        prop = 1, replace = T
      ) %>% 
      ungroup %>% 
      count(dataset, short, result) %>% 
      pivot_wider(
        id_cols = c(dataset, short),
        names_from = result,
        values_from = n,
        values_fill = 0
      ) %>% 
      mutate(
        n = FN + FP + TP,
        Recall = ifelse((TP + FN) == 0, 1, TP / (TP + FN)),
        Precision = ifelse((TP + FP) == 0, 1, TP / (TP + FP)),
        F1 = 2 * Precision * Recall / (Precision + Recall)
      ) %>% 
      select(dataset, short, n, Recall, Precision, F1)
  }
  
  tibble(
    boot_i = seq(boot.n)
  ) %>% 
    mutate(
      res = future_map(boot_i, ~calc_stats_(full_data), .progress = show_progress(), .options = furrr_options(seed = T))
    ) %>% 
    unnest(res) %>%
    group_by(dataset, short) %>%
    summarize(
      n = mean(n),
      Recall_lower = quantile(Recall, 0.025),
      Recall_q1 = quantile(Recall, 0.25),
      Recall_median = median(Recall),
      Recall_q3 = quantile(Recall, 0.75),
      Recall_upper = quantile(Recall, 0.975),
      Recall = mean(Recall),
      Precision_lower = quantile(Precision, 0.025),
      Precision_q1 = quantile(Precision, 0.25),
      Precision_median = median(Precision),
      Precision_q3 = quantile(Precision, 0.75),
      Precision_upper = quantile(Precision, 0.975),
      Precision = mean(Precision),
      F1_lower = quantile(F1, 0.025),
      F1_q1 = quantile(F1, 0.25),
      F1_median = median(F1),
      F1_q3 = quantile(F1, 0.75),
      F1_upper = quantile(F1, 0.975),
      F1 = mean(F1),
      .groups = "drop"
    )
}

# Compare backbone sizes / Deployment models
plan(multisession, workers = availableCores() - 2)
set.seed(18125)

tibble(
  files = list.files("data", pattern = "csv$", full.names = T)
) %>% 
  filter(str_detect(files, "compare_backbone_sizes_._full")) %>% 
  mutate(
    model_size = str_extract(files, "(?<=compare_backbone_sizes_).(?=_full)") %>% 
      str_remove("compare_backbone_sizes_") %>% 
      factor(c("L", "M", "S", "N")),
    new_data = map(files, ~recompute_stats(.x, size_threshold=32), .progress = if (show_progress()) "Recomputing compare backbone sizes" else F)
  ) %>%
  unnest(new_data) %>%
  select(!files) %>%
  write_csv("data/compare_backbone_sizes_combined_recomputed.csv")

plan(sequential)

# Leave-one-out
plan(multisession, workers = availableCores() - 2)
set.seed(91242)

tibble(
  files = list.files("data", pattern = "csv$", full.names = T)
) %>% 
  filter(str_detect(files, "fb_leave_one_out")) %>% 
  mutate(
    new_data = map(files, ~recompute_stats(.x, size_threshold=32), .progress = if (show_progress()) "Recomputing leave-one-out" else F)
  ) %>% 
  mutate(
    leave_out_dataset = str_extract(files, "fb_leave_one_out_(fine_tune_){0,1}(fb_leave_one_out_){0,1}[^_]+") %>% 
      str_remove("fb_leave_one_out_(fine_tune_){0,1}(fb_leave_one_out_){0,1}") %>% 
      str_remove(".csv"),
    fine_tune = str_detect(files, "fine_tune") %>% 
      ifelse("After", "Before") %>% 
      factor(c("Before", "After")),
    leave_out_short = short_name(leave_out_dataset)
  ) %>% 
  select(!files) %>%
  unnest(new_data) %>% 
  mutate(
    across(c(leave_out_short, short), ~factor(.x, sort(unique(.x[!is.na(.x)]))))
  ) %>% 
  relocate(fine_tune, dataset, leave_out_dataset, short, leave_out_short, n, Recall, Precision, F1) %>% 
  write_csv("data/leave_one_out_combined_recomputed.csv")
  
plan(sequential)

# Leave-two-out
plan(multisession, workers = availableCores() - 2)
set.seed(56215)

tibble(
  files = list.files("data", pattern = "csv$", full.names = T)
) %>% 
  filter(str_detect(files, "fb_leave_two_out")) %>% 
  mutate(
    datasets = files %>% 
      str_remove_all("^data/fb_leave_two_out_|\\.csv$") %>% 
      str_split("_", 2, T) %>% 
      set_colnames(c("left", "right")) %>% 
      as_tibble %>% 
      mutate(
        left = ifelse(left == "FULL", "", left),
        left_short = left,
        right_short = right,
        across(contains("short"), short_name)
      )
  ) %>% 
  unnest(datasets) %>% 
  mutate(
    new_data = map(files, ~recompute_stats(.x, size_threshold=32), .progress = if (show_progress()) "Recomputing leave-two-out" else F)
  ) %>% 
  select(!files) %>% 
  unnest(new_data) %>% 
  mutate(
    across(c(left_short, right_short, short), ~factor(.x, sort(unique(.x[!is.na(.x)]))))
  ) %>% 
  relocate(left, left_short, right, right_short, dataset, short, n, Recall, Precision, F1) %>% 
  write_csv("data/leave_two_out_combined_recomputed.csv")
  
plan(sequential)
