library(tidyverse)
library(extrafont)
library(magrittr)
library(ggpubr)

theme_set(
  ggpubr::theme_pubr(
    base_family = "CMU Serif",
    legend = "right"
  ) +
    theme(
      title = element_text(face = "bold", size = 16),
      plot.title = element_text(face = "plain", size = 20, hjust = 0.5),
      panel.grid.major = element_line(color = "gray65", linewidth = 0.25)
    )
)

source("flatbug_palette.R")

# Load data
data <- read_csv("combined_results.csv")

# Reformat data
data <- data %>%
  mutate(
    model = str_extract(model, "(?<=fb_compare_backbone_sizes_).") %>%
      factor(levels = c("N", "S", "M", "L")),
    f1 = 2 * precision * recall / (precision + recall)
  ) %>%
  rename_with(str_to_sentence, c(precision, recall, f1)) %>%
  pivot_longer(c(Precision, Recall, F1), names_to = "metric", values_to = "value")

weighted.var <- function(x, w) {
  sum(w * (x - weighted.mean(x, w))^2) / sum(w)
}

boot_weighted.mean.quantiles <- function(x, w, n, q = c(0.025, 0.975)) {
  prob <- w / sum(w)
  obs <- sapply(1:n, function(i) {
    ind <- sample.int(length(x), replace = TRUE, prob = prob)
    mean(x[ind])
  })

  c("mean" = mean(obs), quantile(obs, q)) %>%
    t() %>%
    as_tibble()
}

data %>%
  group_by(model, metric) %>%
  summarise(
    mean_hilow = boot_weighted.mean.quantiles(
      value,
      log10(n_instances),
      10000,
      c(0.025, 0.25, 0.5, 0.75, 0.975)
    ),
    time = (first(time) / sum(n_instances)) * 1000,
    .groups = "drop"
  ) %>%
  unnest_wider(mean_hilow) %>%
  ggplot(aes(
    x = time,
    y = `50%`,
    ymin = `2.5%`,
    lower = `25%`,
    middle = `50%`,
    upper = `75%`,
    ymax = `97.5%`,
    fill = model,
    group = model
  )) +
  geom_boxplot(stat = "identity") +
  scale_y_continuous(labels = scales::label_percent(), expand = expansion()) +
  scale_fill_flatbug() +
  facet_wrap(~metric, scales = "free_y") +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(
    y = "Metric value",
    x = "Time per instance (ms)"
  )
