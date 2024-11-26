source("flatbug_init.R")

cutoff_data <- read_csv2("data/compare_backbone_sizes_L_full.csv") %>%
  mutate(
    across(c(conf1, conf2, IoU), as.numeric),
    instance_area = ifelse(idx_1 != -1, contourArea_1, contourArea_2),
    dataset = str_extract(image, "[^_]+"),
    short = short_name(dataset)
  ) %>%
  select(dataset, short, instance_area, idx_1, idx_2, conf2, IoU) %>%
  mutate(
    result = case_when(
      idx_1 != -1 & idx_2 == -1 ~ "FN",
      idx_1 == -1 & idx_2 != -1 ~ "FP",
      idx_1 != -1 & idx_2 != -1 ~ "TP"
    )
  ) %>%
  select(!c(idx_1, idx_2)) %>%
  filter(instance_area < 10^6)

cutoff_plot <- cutoff_data %>%
  mutate(
    instance_sqrt_area = sqrt(instance_area)
  ) %>%
  group_by(dataset) %>%
  arrange(instance_sqrt_area) %>%
  mutate(
    frac_at_cs32 = ifelse(
      row_number() == which.min(abs(instance_sqrt_area - 32)),
      cumsum(rep(1 / n(), n())),
      NA_real_
    )
  ) %>%
  ungroup() %>%
  ggplot(aes(instance_sqrt_area)) +
  geom_histogram(
    aes(fill = result, color = after_scale(fill)),
    position = "fill",
    binwidth = 0.035,
    key_glyph = draw_key_point
  ) +
  stat_ecdf(
    geom = "step",
    color = "black",
    # binwidth = 0.07,
    # position = position_nudge(x = -0.07/2)
  ) +
  geom_hline(
    aes(yintercept = frac_at_cs32)
  ) +
  geom_vline(
    xintercept = 32,
    color = "black",
    linewidth = 0.75
  ) +
  scale_x_log10(limits = c(10, 1000), n.breaks = 6) +
  scale_y_continuous(labels = scales::percent_format()) +
  scale_fill_flatbug("simple") +
  coord_cartesian(ylim = 0:1, expand = FALSE) +
  guides(
    fill = guide_legend(override.aes = list(shape = 21, size = 8, stroke = 1, color = "black"))
  ) +
  labs(fill = NULL, x = "Instance size (√px)", y = "Proportion of instances") +
  facet_wrap(~short, ncol = 4) +
  theme(
    panel.background = element_rect(fill = "gray75"),
    panel.spacing.x = unit(1.5, "lines"),
    aspect.ratio = 0.5
  )

ggsave(
  "figures/justify_cutoff32.pdf", cutoff_plot,
  width = 6.75, height = 5.5, scale = 2.15,
  dev = cairo_pdf
)
