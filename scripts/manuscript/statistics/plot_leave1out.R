source("flatbug_init.R")

findoutlier <- function(x, w = NULL, na.rm = TRUE) {
  na_action <- if (na.rm) na.omit else na.fail
  # quantreg::rq(y ~ 1, weights = weight, data = data, tau = qs)
  if (is.null(w)) {
    q1 <- quantile(x, 0.25, na.rm = na.rm)
    q3 <- quantile(x, 0.75, na.rm = na.rm)
  } else {
    q1 <- quantreg::rq(x ~ 1, weights = w, tau = 0.25, na.action = na_action)
    q3 <- quantreg::rq(x ~ 1, weights = w, tau = 0.75, na.action = na_action)
    q1 <- coef(q1)
    q3 <- coef(q3)
  }
  iqr <- q3 - q1
  lower <- q1 - 1.5 * iqr
  upper <- q3 + 1.5 * iqr
  outlier <- (x < lower | x > upper)
  # hinge_min <- min(x[!outlier], na.rm = na.rm)
  # hinge_max <- max(x[!outlier], na.rm = na.rm)
  # outlier & (x < hinge_min | x > hinge_max)
  return(outlier)
}

leave_one_out_data <- "data/leave_one_out_combined_recomputed.csv" %>%
  read_csv() %>%
  filter(leave_out_dataset != "AMI-traps") %>%
  mutate(
    across(c(leave_out_short, short), ~ factor(.x, sort(unique(.x[!is.na(.x)])))),
    fine_tune = factor(fine_tune, levels = c("Before", "After"))
  )

leave_one_out_heatmap <- leave_one_out_data %>%
  pivot_longer(
    cols = c(Recall, Precision, F1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(metric, dataset, fine_tune) %>%
  mutate(
    value = (value - value[leave_out_dataset == "FULL"])
  ) %>%
  drop_na() %>%
  ggplot(aes(leave_out_short, short, fill = value)) +
  geom_tile() +
  scale_fill_flatbug_c(
    "RdWiBu",
    limits = c(-1, 1),
    na.value = "black",
    oob = scales::oob_censor,
    labels = scales::percent_format(),
    expand = expansion(),
    n.breaks = 8
  ) +
  facet_grid(cols = vars(metric), rows = vars(fine_tune), scales = "free") +
  theme(
    axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
    aspect.ratio = 1
  ) +
  guides(
    fill = guide_colourbar(
      title.position = "top",
      title.hjust = 0.5,
      theme = theme(
        legend.key.height = unit(45, "lines"),
        legend.key.width = unit(2.5, "lines")
      )
    )
  ) +
  labs(
    x = "Leave-out dataset",
    y = "Reference Dataset",
    fill = "ΔMetric"
  ) +
  theme(
    axis.title = element_text(size = 20),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),
  )

ggsave(
  "figures/leave_one_out_heatmap.pdf", leave_one_out_heatmap,
  width = 8, height = 5,
  device = cairo_pdf, scale = 2.25, antialias = "subpixel"
)

leave_one_out_metrics <- leave_one_out_data %>%
  filter(dataset == leave_out_dataset | leave_out_dataset == "FULL") %>%
  mutate(
    type = case_when(
      leave_out_dataset == "FULL" ~ "Full",
      fine_tune == "After" ~ "After",
      TRUE ~ "Before"
    ) %>%
      factor(levels = c("Before", "After", "Full"))
  ) %>%
  pivot_longer(
    cols = c(Precision, Recall, F1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(fine_tune, dataset, metric) %>%
  mutate(
    full = value[type == "Full"]
  ) %>%
  ungroup() %>%
  filter(type != "Full") %>%
  select(!fine_tune) %>%
  group_by(metric, type) %>%
  arrange(value) %>%
  mutate(
    weight = n / mean(n),
    outlier = findoutlier(value, weight),
    outlier_dir = rep(0, n()) %>%
      magrittr::inset(outlier, rep_len(c(-0.25, 1.25), sum(outlier)))
  ) %>%
  ungroup() %>%
  ggplot(aes(metric, value, fill = type, weight = weight)) +
  geom_violin(
    scale = "width",
    trim = FALSE,
    bounds = 0:1,
    width = 0.5,
    color = NA,
    # alpha = 0.25,
    position = position_dodge(width = 0.75),
    show.legend = FALSE
  ) +
  scale_fill_flatbug("RdBu", lighten = 0.75) +
  ggnewscale::new_scale_fill() +
  geom_boxplot(
    aes(fill = type),
    linewidth = 0.75,
    # key_glyph = draw_key_point,
    outlier.size = 1,
    color = "black"
  ) +
  geom_text(
    aes(
      label = ifelse(outlier, as.character(short), NA_character_),
      group = type,
      hjust = outlier_dir
    ),
    position = position_dodge(width = 0.75),
    family = "mono",
    fontface = "bold",
    size = 3,
    na.rm = TRUE
  ) +
  scale_fill_flatbug("RdBu") +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 1),
    limits = 0:1,
    breaks = seq(0, 1, 0.1),
    minor_breaks = seq(0, 1, 0.05),
    expand = expansion(0, 0)
  ) +
  scale_x_discrete(labels = str_to_sentence) +
  facet_wrap(~metric, scales = "free_x") +
  labs(y = NULL, x = NULL, fill = "Fine-tuning") +
  # guides(
  #   fill = guide_legend(
  #     override.aes = list(
  #       shape = 21,
  #       color = "black",
  #       stroke = 1,
  #       size = 8
  #     )
  #   )
  # ) +
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.line.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.y = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed"),
    legend.key.height = unit(10, "lines"),
    legend.key.width = unit(2.5, "lines")
  )

ggsave(
  "figures/leave_one_out_metrics.pdf", leave_one_out_metrics,
  width = 5, height = 4, scale = 2.25,
  device = cairo_pdf, antialias = "subpixel"
)