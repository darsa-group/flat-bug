leave_one_out_data <- "data/leave_one_out_combined_recomputed.csv" %>% 
  read_csv(show_col_types = F) %>% 
  filter(leave_out_dataset != "AMI-traps") %>% 
  mutate(
    across(c(leave_out_short, short), ~factor(.x, sort(unique(.x[!is.na(.x)])))),
    fine_tune = factor(fine_tune, levels = c("Before", "After"))
  ) 

l1o_heatmap_plt <- leave_one_out_data %>% 
  pivot_longer(
    cols = c(Recall, Precision, F1),
    names_to = "metric",
    values_to = "value"
  ) %>%
  group_by(metric, dataset, fine_tune) %>% 
  mutate(
    value = (value - value[leave_out_dataset == "FULL"])
  ) %>% 
  drop_na %>% 
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
  "figures/leave_one_out_heatmap.pdf", 
  l1o_heatmap_plt,
  device = cairo_pdf,
  width = 8, height = 5, 
  scale = 2.25, 
  antialias = "subpixel"
)

l1o_easy_treat <- function(x) {
  x %>% 
    as.character %>% 
    case_match(
      "Before" ~ "Out-of-box",
      "After" ~ "Fine-tuned",
      "Full" ~ "Full"
    )
}

leave_one_out_cleaned <- leave_one_out_data %>% 
  filter(dataset == leave_out_dataset | leave_out_dataset == "FULL") %>% 
  mutate(
    type = ifelse(leave_out_dataset == "FULL", "Full", ifelse(fine_tune == "After", "After", "Before")) %>% 
      factor(c("Before", "After", "Full")),
    type = factor(l1o_easy_treat(type), l1o_easy_treat(levels(type)))
  ) %>% 
  select(fine_tune, dataset, short, type, n, Precision, Recall, F1) %>% 
  pivot_longer(
    cols = c(Precision, Recall, F1),
    names_to = "metric",
    values_to = "value"
  )  %>% 
  mutate(
    full = value[type == "Full"],
    rel_delta = (value - full)/ifelse(value >= full, 1 - full, full),
    rel_delta = ifelse(value == full, 0, rel_delta),
    .by = c(fine_tune, dataset, metric)
  ) %>%  
  filter(type != "Full") %>% 
  select(!fine_tune) 

leave_one_out_plot_elems <- list(
  scale_fill_flatbug("RdBu", lighten = 0.33),
  geom_label(
    aes(
      label = ifelse(outlier, as.character(short), NA_character_),
      group = type,
      hjust = outlier_dir
    ),
    position = position_dodge(width = 0.75),
    family = "mono", fontface = "bold",
    size = 3, fill = "white", label.size = 0,
    na.rm = T
  ),
  scale_x_discrete(labels = str_to_sentence),
  facet_wrap(~metric, scales = "free_x"),
  labs(y = "Normalized relative change (δ)", x = NULL, fill = NULL),
  guides(
    fill = guide_legend(
      override.aes = list(color = "black")
    )
  ),
  theme(
    axis.ticks.x = element_blank(),
    axis.text.x = element_blank(),
    axis.line.x = element_blank(),
    panel.grid.major.y = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.y = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed"),
    legend.key.height = unit(10, "lines"),
    legend.key.width = unit(2.5, "lines"),
    legend.text = element_text(hjust = 1)
  )
)

l1o_delta_summary <- leave_one_out_cleaned %>%
  summarize(
    delta = weighted_cl_boot(rel_delta, n),
    dbound = rel_delta[which.max(abs(rel_delta) * ((if (delta$pval <= 0.05) sign(delta$y) else 1) == sign(rel_delta)))],
    .by = c(metric, type)
  ) %>% 
  unnest(delta) %>% 
  mutate(
    ci_el = pmap(list(ymin, y, ymax), function(...) confint_element(c(...), "edge", c(0, 0.1), scales::label_percent(.1), c(" ", ", "))),
    p = label_pvalue(pval, 0.001, "*")
  ) %>% 
  unnest(ci_el) %>% 
  mutate(
    label = str_c(label, " ", str_extract(p, "[^ \\d]+$"))
  )

l1o_f1_plt <- leave_one_out_cleaned %>%
  group_by(metric, type) %>% 
  arrange(rel_delta) %>% 
  mutate(
    weight = log(n),
    weight = weight / mean(weight),
    outlier = findoutlier(rel_delta, weight),
    outlier_dir = rep(0, n()) %>%
      magrittr::inset(outlier, rep_len(c(-0.25, 1.25), sum(outlier)))
  ) %>% 
  ungroup %>% 
  ggplot(aes(metric, rel_delta, fill = type, weight = weight)) +
  geom_violin(
    scale = "width", trim = F,
    bounds = c(-1, 1),
    bw = 0.05,
    color = NA,
    width = 0.5, 
    position = position_dodge(width = 0.75),
    key_glyph = draw_key_boxplot
  ) +
  geom_pointrange(
    data = l1o_delta_summary,
    aes(metric, y, ymin = ymin, ymax = ymax),
    inherit.aes = F,
    position = position_dodge2(0.75)
  ) +
  scale_y_continuous(
    labels = scales::label_percent(accuracy = 1), 
    limits = c(-1.1, 1.1), 
    breaks = seq(-1, 1, 0.2),
    minor_breaks = seq(-1, 1, 0.1),
    expand = expansion(0, 0.05)
  ) +
  leave_one_out_plot_elems + 
  geom_label(
    data = l1o_delta_summary,
    aes(
      x = metric, 
      y = pmin(1.05, pmax(-1.05, 0.2 * sign(dbound) + dbound)), 
      label = label, 
      vjust = ifelse(sign(y) == 1, 0.25, 0.75)
    ),
    inherit.aes = F,
    family = "CMU Serif", fontface = "bold", size = 4,
    label.size = 0.5, label.padding = unit(0.5, "lines"),
    hjust = 0.5, position = position_dodge2(0.75)
  )

ggsave(
  "figures/leave_one_out_f1.pdf", 
  l1o_f1_plt, 
  device = cairo_pdf,
  width = 6, height = 4, 
  scale = 2.25, 
  antialias = "subpixel"
)

l1o_delta_summary_latex <- l1o_delta_summary %>% 
  select(metric, type, label) %>% 
  mutate(
    metric = case_match(
      metric,
      "Precision" ~ "P",
      "Recall" ~ "R",
      "F1" ~ "F1"
    ),
    type = case_match(
      type,
      "Out-of-box" ~ "oob",
      "Fine-tuned" ~ "ft"
    ),
    label = label %>% 
      str_replace_all("%", "\\\\%") %>% 
      str_remove("\\s\\S+$"),
    ltx = str_c("\\defexperiment{2}{",metric,"-", type, "}{", label, "}")
  ) %>% 
  arrange(metric, type) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n")

add_group("Experiment 2 - Leave-one-out summary")
write_data("Experiment 2 - Leave-one-out summary", l1o_delta_summary_latex)

l1o_delta_stratified_latex <- leave_one_out_cleaned %>% 
  select(short, metric, type, rel_delta) %>% 
  mutate(
    metric = case_match(
      metric,
      "Precision" ~ "P",
      "Recall" ~ "R",
      "F1" ~ "F1"
    ),
    type = case_match(
      type,
      "Out-of-box" ~ "oob",
      "Fine-tuned" ~ "ft"
    ),
    label = scales::label_percent(.1)(rel_delta) %>% 
      str_replace_all("%", "\\\\%"),
    ltx = str_c("\\defexperiment{2}{",metric,"-", type, "-", short,"}{", label, "}")
  ) %>% 
  arrange(metric, type, short) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n") 

add_group("Experiment 2 - Leave-on-out stratified")
write_data("Experiment 2 - Leave-on-out stratified", l1o_delta_stratified_latex)
