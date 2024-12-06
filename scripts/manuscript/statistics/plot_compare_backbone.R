source("flatbug_init.R")

quantile_ranges <- function(q) {
  ord <- order(q)
  q <- q[ord]
  n <- length(q)
  lag_q <- lag(q, default = 0)[1:n]
  lead_q <- lead(q, default = 1)[1:n]

  qr <- (q - lag_q) * c(1, rep(0.5, n - 1)) +
    (lead_q - q) * c(rep(0.5, n - 1), 1)

  qr[order(ord)]
}

compare_sizes_data <- "data/compare_backbone_sizes_combined_recomputed.csv" %>%
  read_csv() %>%
  mutate(
    dataset = str_remove(dataset, "^01-partial-"),
    model_size = factor(model_size, levels = c("L", "M", "S", "N"))
  )

compare_sizes_metrics <- compare_sizes_data %>%
  arrange(short, dataset) %>%
  mutate(
    across(c(short, dataset), ~ factor(.x, unique(.x))),
    model_size = model_size %>%
      case_match(
        "L" ~ "Large",
        "M" ~ "Medium",
        "S" ~ "Small",
        "N" ~ "Nano"
      ) %>%
      factor(c("Large", "Medium", "Small", "Nano"))
  ) %>%
  pivot_longer(!c(model_size, dataset, short, n), names_to = "metric_t", values_to = "value") %>%
  mutate(
    metric = str_extract(metric_t, "^[^_]+"),
    metric_t = ifelse(
      metric_t == metric, "mean", str_remove(metric_t, str_c(metric, "_*"))
    )
  ) %>%
  filter(metric_t %in% c("lower", "q1", "median", "q3", "upper")) %>%
  group_by(model_size, dataset, metric) %>%
  mutate(
    q = case_when(
      metric_t == "lower" ~ 0.025,
      metric_t == "q1" ~ 0.25,
      metric_t == "median" ~ 0.5,
      metric_t == "q3" ~ 0.75,
      metric_t == "upper" ~ 0.975
    )
  ) %>%
  mutate(
    weight = n * quantile_ranges(q)
  ) %>%
  select(!q) %>%
  group_by(model_size, metric) %>%
  mutate(
    weight = weight / mean(weight)
  ) %>%
  ungroup() %>%
  arrange(model_size, dataset, metric) %>%
  group_by(model_size, metric) %>%
  reframe(
    value = Hmisc::wtd.quantile(
      x = value,
      weights = weight,
      probs = c(0.025, 0.25, 0.5, 0.75, 0.975),
      type = "i/(n+1)"
    ),
    quantile = c(0.025, 0.25, 0.5, 0.75, 0.975)
  ) %>%
  mutate(
    qname = case_when(
      quantile == 0.025 ~ "lower",
      quantile == 0.25 ~ "q1",
      quantile == 0.5 ~ "median",
      quantile == 0.75 ~ "q3",
      quantile == 0.975 ~ "upper"
    )
  ) %>%
  select(!quantile) %>%
  pivot_wider(id_cols = c(model_size, metric), names_from = qname, values_from = value) %>%
  arrange(model_size, metric) %>%
  mutate(
    grp = paste0(model_size, metric) %>%
      factor(rev(unique(.)))
  ) %>%
  ggplot(
    aes(
      median,
      metric,
      fill = model_size,
      label = scales::percent_format(0.1)(median),
      group = grp
    )
  ) +
  geom_boxplot(
    aes(
      xmiddle = median,
      xlower = q1,
      xmin = lower,
      xupper = q3,
      xmax = upper
    ),
    stat = "identity",
    width = .75,
    position = position_dodge(width = 1),
    key_glyph = draw_key_point
  ) +
  geom_text(
    aes(
      x = q3
    ),
    position = position_dodge(width = 1),
    size = 5,
    hjust = -0.1,
    vjust = -0.15,
    fontface = "bold",
    family = "CMU Serif"
  ) +
  annotation_custom(grid::linesGrob(y = c(0, 0), gp = grid::gpar(lwd = 0.5))) +
  scale_x_continuous(
    limits = c(0.4, 1),
    breaks = seq(0, 1, 0.1),
    minor_breaks = seq(0, 1, 0.05),
    expand = expansion(0, c(0, 0.015)),
    labels = scales::percent_format()
  ) +
  scale_fill_flatbug() +
  scale_color_flatbug("main_text", guide = "none") +
  facet_wrap(~metric, scales = "free_y", strip.position = "left", ncol = 1) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, fill = "Model\nSize") +
  guides(
    fill = guide_legend(
      override.aes = list(
        shape = 21,
        size = 10,
        stroke = 1.25,
        color = "black"
      )
    )
  ) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    # legend.box.spacing = unit(1.5, "lines"),
    panel.grid.major.x = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.x = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed"),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm")
  )

ggsave(
  "figures/compare_sizes_metrics.pdf", compare_sizes_metrics,
  width = 5, height = 2, scale = 2.5,
  device = cairo_pdf, antialias = "subpixel"
)

compare_sizes_stratified <- compare_sizes_data %>%
  arrange(F1) %>%
  mutate(
    short = factor(short, rev(unique(short))),
    dataset = factor(dataset, rev(unique(dataset)))
  ) %>%
  pivot_longer(
    cols = c(Recall, Precision, F1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(metric, short) %>%
  mutate(
    min_val = ifelse(row_number() == which.min(value), min(value), NA_real_),
    max_val = ifelse(row_number() == which.max(value), max(value), NA_real_)
  ) %>%
  ungroup() %>%
  ggplot(aes(value, dataset, fill = model_size)) +
  geom_text(
    aes(x = min_val - 0.01, label = short),
    hjust = 1,
    fontface = "bold",
    family = "CMU Serif"
  ) +
  geom_line(aes(group = short), linewidth = 0.75) +
  geom_point(shape = 21, color = "black", stroke = 1, size = 4) +
  scale_fill_flatbug() +
  scale_x_continuous(
    labels = scales::percent_format(accuracy = 1),
    limits = 0:1,
    expand = expansion(0, c(0, 0.01)),
    breaks = seq(-1, 2, 0.1),
    minor_breaks = seq(-1, 2, 0.05)
  ) +
  scale_y_discrete(position = "right") +
  facet_wrap(~metric, ncol = 1) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, fill = "Model\nsize") +
  guides(
    fill = guide_legend(override.aes = list(size = 8))
  ) +
  theme(
    panel.grid.major.x = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.x = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed")
  )

ggsave(
  "figures/compare_sizes_stratified.pdf", compare_sizes_stratified,
  width = 6, height = 5, scale = 3,
  device = cairo_pdf, antialias = "subpixel"
)
