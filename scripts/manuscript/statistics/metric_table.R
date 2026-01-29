suppressPackageStartupMessages({
  library(kableExtra)
})

max_to_bold_min_to_italic <- function(x) {
  if (any(is.na(x))) return(as.character(x))
  ma <- which(x == max(x))
  mi <- which(x == min(x))
  x[ma] <- paste0("\\textbf{", x[ma], "}")
  x[mi] <- paste0("\\textit{", x[mi], "}")
  x
}

backbone_size_results <- read_csv("data/backbone_size_results.csv")
AP_results <- read_csv("data/AP_results.csv")

nice_colnames <- c(
  "Backbone size" = "model_size",
  "Metric" = "metric",
  "$Q_{2.5\\%}$" = "lower",
  "$Q_{25\\%}$" = "q1",
  "$Q_{50\\%}$" = "median",
  "$Q_{75\\%}$" = "q3",
  "$Q_{97.5\\%}$" = "upper"
)


metrics_combined <- AP_results %>% 
  mutate(
    model_size = model %>% 
      case_match(
        "L" ~ "Large",
        "M" ~ "Medium",
        "S" ~ "Small",
        "N" ~ "Nano"
      )
  ) %>% 
  select(!model) %>% 
  pivot_longer(!model_size, names_to = "metric", values_to = "median") %>% 
  mutate(
    metric = metric %>% 
      str_replace("_", "-") %>% 
      paste0("\\textsuperscript{*}")
  ) %>% 
  bind_rows(
    backbone_size_results
  ) %>% 
  relocate(
    median, .after = q1
  ) %>% 
  mutate(
    model_size = model_size %>% 
      factor(c("Nano", "Small", "Medium", "Large"))
  ) %>% 
  arrange(metric, desc(model_size))

row_index <- metrics_combined$metric %>% 
  factor(.) %>% 
  as.numeric %>% 
  diff %>% 
  {c(0,.)} %>% 
  as.logical %>% 
  cumsum %>% 
  table %>% 
  set_names(unique(metrics_combined$metric))

quantiles_latex <- paste0(names(nice_colnames)[3:7], collapse=", ")
section_reference <- "\\sref{sec:res_exp1}"

metric_table_latex <- do.call(rename, c(list(metrics_combined), as.list(nice_colnames))) %>% 
  group_by(Metric) %>% 
  mutate(
    across(!`Backbone size`, ~round(.x, 3)),
    across(!`Backbone size`, max_to_bold_min_to_italic),
    across(!`Backbone size`, ~replace_na(.x, "-"))
  ) %>% 
  ungroup %>% 
  select(!Metric) %>% 
  kable(
    "latex",
    escape = F,
    booktabs = T,
    digits = 3,
    align = c("rccccc"),
    caption = c(
      "Quantile summary ({quantiles_latex}) of the bootstrap distributions for all",
      "evaluation metrics in Experiment 1, stratified by YOLOv8 backbone size",
      "(Large, Medium, Small, Nano). The table provides the exact numeric values",
      "underlying the performance comparisons discussed in {section_reference},",
      "including median performance and lower/upper confidence quantiles for",
      "AP50, AP50–95, F1, Precision, and Recall."
    ) %>%
      paste(collapse = " ") %>% 
      str_glue
  ) %>%
  column_spec(
    2:6,
    width = "1.5cm"
  ) %>% 
  pack_rows(
    index = row_index,
    escape = F
  ) %>%
  add_footnote(
    "For simplicity confidence intervals are not computed for AP50 and AP50-95,
    instead the values reported for these in the column $Q_{50\\%}$ is not the 
    median, but rather just the raw metric value computed directly on the full 
    evaluation output.",
    "symbol",
    T,
    escape = F
  ) %>% 
  str_split_1("\n") %>% 
  {
    .[1] <- paste0(.[1], "[htb]")
    .[2] <- "\\centering"
    .
  } %>% 
  {
    reord <- .[!str_detect(., "caption\\{")]
    e3pt <- which(str_detect(reord, "end\\{threeparttable\\}")) + 1
    c(reord[1:(e3pt-1)], .[str_detect(., "caption\\{")], reord[e3pt:length(reord)])
  } %>% 
  {
    c(.[1:(length(.)-1)], "\\label{tab:metrics_tab}", .[length(.)])
  } %>% 
  paste0(collapse = "\n") 

add_group("Experiment 1 - Backbone-size table")
write_data("Experiment 1 - Backbone-size table", latex_env2macro(metric_table_latex, "MetricsTable"))
