compare_sizes_data <- "data/compare_backbone_sizes_combined_recomputed.csv" %>% 
  read_csv(show_col_types = F) %>% 
  mutate(
    dataset = str_remove(dataset, "^01-partial-"),
    model_size = factor(model_size, levels = c("L", "M", "S", "N"))
  )

backbone_size_results <- compare_sizes_data %>% 
  arrange(short, dataset) %>% 
  mutate(
    across(c(short, dataset), ~factor(.x, unique(.x))),
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
  group_by(model_size, metric) %>% 
  mutate(
    q = case_when(
      metric_t == "lower" ~ 0.025,
      metric_t == "q1" ~ 0.25,
      metric_t == "median" ~ 0.5,
      metric_t == "q3" ~ 0.75,
      metric_t == "upper" ~ 0.975
    ),
    weight = log(n) * quantile_ranges(q)
  ) %>% 
  reframe(
    value = Hmisc::wtd.quantile(value, weight/mean(weight), probs = c(0.025, 0.25, 0.5, 0.75, 0.975), type = "i/(n+1)"),
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
  pivot_wider(id_cols = c(model_size, metric), names_from = qname, values_from = value)

backbone_results_latex <- backbone_size_results %>% 
  pivot_longer(!c(model_size, metric), names_to = "qtd") %>% 
  mutate(
    qtd = if_else(qtd == "median", "", qtd) %>% 
      factor(c("lower", "q1", "", "q3", "upper", "ci")),
    model_size = case_match(
      model_size,
      "Large" ~ "large",
      "Medium" ~ "medium",
      "Small" ~ "small",
      "Nano" ~ "nano"
    ),
    label = scales::label_percent(.1)(value),
    label = label %>% 
      str_replace_all("%", "\\\\%") %>% 
      str_remove("\\s\\S+$"),
  ) %>% 
  select(!value) %>% 
  nest(dat = !c(model_size, metric)) %>%
  mutate(
    dat = map(dat, function(d) {
      d %>% 
        arrange(qtd) %>% 
        add_row(
          qtd = "ci", label = str_c(.$label[3], " [", .$label[1], ", ", .$label[5], "]")
        )
    })
  ) %>% 
  unnest(dat) %>% 
  mutate(
    ltx_resname = str_c(model_size, "-", metric, "-", qtd) %>% 
      str_remove("-$"),
    ltx = str_c("\\defexperiment{1}{", ltx_resname, "}{", label, "}")
  ) %>% 
  arrange(model_size, metric, qtd) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n")

add_group("Experiment 1 - Backbone-size metrics")
write_data("Experiment 1 - Backbone-size metrics", backbone_results_latex)

comp_bb_plt <- backbone_size_results %>%
  mutate(
    metric = factor(metric, c("F1", "Precision", "Recall"))
  ) %>% 
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
  geom_segment(
    aes(
      y=(2 + 4 - as.numeric(model_size))/4 + 0.03, yend=(3 + 4 - as.numeric(model_size))/4 - 0.03,
      color=model_size
    ),
    linewidth = 1.25
    # position = position_dodge(width = 1)
  ) +
  geom_label(
    aes(
      x = upper
    ),
    position = position_dodge(width = 1),
    size = 5,
    hjust = 0,
    vjust = 0.5,
    fontface = "bold",
    family = "CMU Serif",
    fill = "white",
    label.size = 0
  ) +
  annotation_custom(grid::linesGrob(y = c(0, 0), gp = grid::gpar(lwd = 2, col = "gray50"))) +
  scale_x_continuous(
    limits = c(0.75, 1),
    breaks = seq(0, 1, 0.05),
    minor_breaks = seq(0, 1, 0.025),
    expand = expansion(0, c(0, 0.015)), 
    labels = scales::percent_format()
  ) +
  scale_fill_flatbug() +
  scale_color_flatbug("main_text", guide = "none") +
  facet_wrap(~metric, scales = "free_y", strip.position = "left", ncol = 1) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, fill = "Model\nsize") +
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
    # legend.box.background = element_blank(),
    # legend.frame = element_blank(),
    # legend.background = element_blank(),
    legend.position = "inside",
    legend.position.inside = c(0.94, 0.8),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    # legend.box.spacing = unit(1.5, "lines"),
    panel.grid.major.x = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.x = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed"),
    plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm")
  ) 

ggsave(
  "figures/compare_sizes_f1.pdf", 
  comp_bb_plt, 
  device = cairo_pdf,
  width = 5, height = 2.5, 
  scale = 2.5, 
  antialias = "subpixel"
)

anno_mat_plt <- magick::image_read("figures/annotation_mosaic.jpg") %>%
  magick::image_sample(2000) %>%
  magick::image_ggplot()
  # theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))
  # theme(plot.margin = margin())

pred_mat_plt <- magick::image_read("figures/prediction_mosaic.jpg") %>%
  magick::image_sample(2000) %>%
  magick::image_ggplot()
  # theme(plot.margin = margin(0.1, 0.1, 0.1, 0.1, "cm"))
  # theme(plot.margin = margin())

comp_mat_plt <- list(
  A = free(comp_bb_plt),
  B = anno_mat_plt,
  C = pred_mat_plt
) %>%
  wrap_plots(
    heights = c(1.5, 1), widths = c(1, 1), design = "AA\nBC",
    axes = "keep",
    tag_level = "keep"
  ) +
  plot_annotation(
    tag_levels = "A", tag_prefix = "", tag_suffix = ")"
  ) &
  theme(
    plot.tag = element_text(family = "CMU Serif", face = "bold", size = 20, vjust = 1), 
    plot.tag.position = "left",
    plot.tag.location = "margin"
  )

ggsave(
  "figures/compare_sizes_mosaic_f1.pdf", 
  comp_mat_plt, 
  device = cairo_pdf,
  width = 5, height = 3.8, 
  scale = 2.5, 
  antialias = "subpixel"
)

## Difference in worst-case performance of medium/small/nano models compared to large

worst_case_results <- compare_sizes_data %>% 
  select(model_size, dataset, short, n, contains("_lower")) %>% 
  pivot_longer(contains("_lower"), names_to = "metric", names_pattern = "(^\\w+)_") %>% 
  group_by(dataset) %>% 
  arrange(model_size) %>% 
  mutate(
    delta = value - value[model_size == "L"]
  ) %>%
  ungroup %>% 
  filter(model_size != "L") %>% 
  summarize(
    delta = weighted_cl_boot(delta, log(n), boot=10000),
    .by = c(model_size, metric)
  ) %>% 
  unnest_wider(delta) %>% 
  mutate(
    ci_el = pmap(list(ymin, y, ymax), function(...) confint_element(c(...), "edge", c(0, 0.1), scales::label_percent(.01), c(" ", ", "))),
    p = label_pvalue(pval, 0.001, "*")
  ) %>% 
  unnest(ci_el) %>% 
  mutate(
    label = str_c(label, " ", str_extract(p, "[^ \\d]+$"))
  )

worst_case_latex <- worst_case_results %>% 
  select(model_size, metric, label) %>% 
  mutate(
    model_size = case_match(
      model_size,
      "M" ~ "medium",
      "S" ~ "small",
      "N" ~ "nano"
    ),
    label = label %>% 
      str_replace_all("%", "\\\\%") %>% 
      str_remove("\\s\\S+$"),
    ltx = str_c("\\defexperiment{1}{", model_size,"-delta-", metric, "}{", label, "}")
  ) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n") 

add_group("Experiment 1 - Backbone-size worst case")
write_data("Experiment 1 - Backbone-size worst case", worst_case_latex)

worst_case_penalty_plt <- worst_case_results %>%
  filter(metric == "F1") %>% 
  ggplot(aes(y, forcats::fct_rev(model_size), xmin = ymin, xmax = ymax, label = label)) +
  geom_pointrange() +
  geom_text(
    position = position_nudge(y = 0.1), 
    family = "CMU Serif", fontface = "bold", 
    size = 4, hjust = 0.5
  ) +
  scale_y_discrete(expand = expansion(0, 0.25)) +
  scale_x_continuous(labels = scales::label_percent(1), breaks = seq(-1, 1, 0.01)) +
  labs(x = "Difference in worst-case performance", y = "Model size")

ggsave(
  "figures/worst_case_size_penalty.pdf", 
  worst_case_penalty_plt,
  device = cairo_pdf,
  width = 4, height = 2,
  scale = 2.25, 
  antialias = "subpixel"
)


## Subdataset stratified performance of different backbone sizes

comb_bb_f1_stratified_plt <- compare_sizes_data %>% 
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
  ungroup %>% 
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
    limits = c(0.4, 1), 
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
    plot.margin = margin(0.1, 0.1, 0.1, 0.5, "cm"),
    panel.grid.major.x = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.x = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed")
  )

ggsave(
  "figures/compare_sizes_stratified.pdf",
  comb_bb_f1_stratified_plt,
  device = cairo_pdf,
  width = 6, height = 4,
  scale = 3, 
  antialias = "subpixel"
)

