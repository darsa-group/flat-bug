source("flatbug_init.R")

model_erda_dir <- c(
  "L" = "L_ucloud_done_V2",
  "M" = "M_ucloud_done_V2",
  "S" = "S_ucloud_done_V2",
  "N" = "N_ucloud_done_V2"
)

model_url <- str_c(
  "https://anon.erda.au.dk/share_redirect/OzZk71Y5SS/",
  model_erda_dir,
  "/compare_backbone_sizes/fb_compare_backbone_sizes_",
  names(model_erda_dir),
  "\\D/results.csv"
) %>%
  set_names(names(model_erda_dir))

get_results <- function(pattern, num_max = 5) {
  res <- tibble(
    url = pattern
  ) %>%
    group_by(url) %>%
    reframe(
      i = seq(num_max),
      url = url %>%
        str_replace("\\\\D", ifelse(i == 1, "", i))
    ) %>%
    ungroup() %>%
    drop_na() %>%
    filter(map_lgl(url, RCurl::url.exists, .progress = "Finding results...")) %>%
    mutate(
      restarts = i - 1,
      data = map(url, read_csv, show_col_types = F, .progress = "Reading results...")
    )

  if (nrow(res) == 0) {
    return(tibble())
  }

  res %>%
    select(!i) %>%
    unnest(data) %>%
    group_by(url) %>%
    arrange(epoch) %>%
    mutate(
      across(matches("val|metrics"), function(x) {
        w <- x == lag(x, default = -9999.1234)
        w[is.na(w)] <- TRUE
        x[w] <- NA
        x
      })
    ) %>%
    ungroup()
}

data <- tibble(
  model = c("N", "S", "M", "L") %>%
    factor(c("N", "S", "M", "L")),
) %>%
  # filter(model %in% names(model_url)) %>%
  mutate(
    data = model %>%
      as.character() %>%
      {
        model_url[.]
      } %>%
      map(get_results)
  ) %>%
  unnest(data)

loss_plt <- data %>%
  select(restarts, model, epoch, contains("loss")) %>%
  mutate(
    `train/mean_loss` = across(matches("train")) %>% rowMeans(),
    `val/mean_loss` = across(matches("val")) %>% rowMeans()
  ) %>%
  pivot_longer(
    contains("loss"),
    names_to = "type",
    names_pattern = "(.+)_loss",
    values_to = "loss"
  ) %>%
  mutate(
    split = str_extract(type, "train|val") %>%
      factor(c("train", "val")),
    metric = str_remove(type, "^[^\\/]+\\/") %>%
      factor(),
    model = model %>%
      {
        c("N" = "Nano", "S" = "Small", "M" = "Medium", "L" = "Large")[.]
      } %>%
      unname() %>%
      factor(c("Nano", "Small", "Medium", "Large"))
  ) %>%
  drop_na() %>%
  filter(epoch > 10) %>%
  # group_by(metric) %>%
  # mutate(
  #   mloss = min(loss, na.rm=T) - 0.01
  # ) %>%
  ggplot(aes(epoch, loss)) +
  geom_point(
    aes(x = NA_real_, y = NA_real_, fill = model),
    size = 5, stroke = 1, shape = 21,
    color = "black"
  ) +
  geom_line(
    aes(x = NA_real_, y = NA_real_, linetype = split),
    linewidth = 1,
    color = "black"
  ) +
  geom_line(
    aes(color = model, linetype = split),
    linewidth = 0.5,
    show.legend = FALSE
  ) +
  scale_y_sqrt(expand = expansion(c(0, 0.1), 0)) +
  scale_color_flatbug() +
  scale_fill_flatbug() +
  facet_wrap(~metric, scales = "free_y", nrow = 2) +
  labs(x = "Epochs", y = "Loss", fill = "Model\nSize", linetype = "\nSplit") +
  guides(
    fill = guide_legend(ncol = 2),
    linetype = guide_legend(ncol = 1, theme = theme(legend.key.width = unit(2.5, "lines")))
  ) +
  theme(
    legend.title = element_text(hjust = 0.5),
    legend.title.position = "top",
    legend.position = "bottom",
    legend.justification = "top",
    legend.box = "horizontal",
    legend.box.just = "top",
    legend.background = element_blank(),
    legend.box.background = element_rect(fill = "#FCFCF5", color = "black", linewidth = 1)
  )

ggsave(
  "figures/train_ loss_plt.pdf", loss_plt,
  width = 5, height = 4, scale = 2.5,
  device = cairo_pdf, antialias = "subpixel"
)


metrics_plt <- data %>%
  select(model, epoch, contains("metrics")) %>%
  mutate(
    `metrics/F1(M)` = f1_from_rp(`metrics/recall(M)`, `metrics/precision(M)`),
    `metrics/F1(B)` = f1_from_rp(`metrics/recall(B)`, `metrics/precision(B)`),
    `metrics/Fitness(M)` = 0.9 * `metrics/mAP50-95(M)` + 0.1 * `metrics/mAP50(M)`,
    `metrics/Fitness(B)` = 0.9 * `metrics/mAP50-95(B)` + 0.1 * `metrics/mAP50(B)`
  ) %>%
  pivot_longer(
    contains("metrics"),
    names_to = "type",
    names_pattern = "metrics/(.+)",
    values_to = "value"
  ) %>%
  mutate(
    task = str_extract(type, "(?<=\\()\\w+(?=\\))") %>%
      {
        c("M" = "Mask", "B" = "Box")[.]
      } %>%
      unname() %>%
      factor(c("Box", "Mask")),
    type = str_extract(type, "[^\\(\\)]+") %>%
      ifelse(. == "precision", "Precision", .) %>%
      ifelse(. == "recall", "Recall", .) %>%
      factor(c("F1", "Recall", "Precision", "mAP50", "mAP50-95", "Fitness")),
    model = model %>%
      {
        c("N" = "Nano", "S" = "Small", "M" = "Medium", "L" = "Large")[.]
      } %>%
      unname() %>%
      factor(c("Nano", "Small", "Medium", "Large"))
  ) %>%
  filter(epoch > 10) %>%
  # filter(epoch > 10 & epoch %% 25 == 1) %>%
  # filter(epoch > 25) %>%
  group_by(model, task, type) %>%
  mutate(
    is_best = row_number() == which.max(value)
  ) %>%
  ungroup() %>%
  arrange(model, task, type, epoch) %>%
  drop_na() %>%
  # filter(type == "F1")
  ggplot(aes(epoch, value)) +
  geom_point(
    aes(color = model, shape = task),
    size = 1,
    show.legend = FALSE
  ) +
  geom_point(
    aes(x = NA_real_, y = NA_real_, fill = model),
    size = 5, stroke = 1, shape = 21,
    color = "black"
  ) +
  geom_line(
    aes(x = NA_real_, y = NA_real_, linetype = task),
    linewidth = 1,
    color = "black"
  ) +
  # geom_vline(xintercept = 50) +
  # geom_point(
  #   aes(ifelse(is_best, epoch, NA), fill = model, shape = task),
  #   stroke = 1, size = 5,
  #   color = "black",
  #   show.legend = FALSE
  # ) +
  geom_smooth(
    aes(color = model, linetype = task),
    linewidth = 0.5,
    show.legend = FALSE,
    method = "gam", se = FALSE, , model.args = list(family = "quasibinomial")
  ) +
  scale_y_continuous(expand = expansion()) +
  scale_color_flatbug() +
  scale_fill_flatbug() +
  scale_shape_manual(values = c(16, 1)) +
  facet_wrap(~type, scales = "free_y", ncol = 3) +
  labs(x = "Epoch", y = NULL, fill = "Model\nSize", linetype = "\nTask") +
  guides(
    fill = guide_legend(ncol = 2),
    linetype = guide_legend(ncol = 1, theme = theme(legend.key.width = unit(2.5, "lines")))
  ) +
  theme(
    legend.title = element_text(hjust = 0.5),
    legend.title.position = "top",
    legend.position = "bottom",
    legend.justification = "top",
    legend.box = "horizontal",
    legend.box.just = "top",
    legend.background = element_blank(),
    legend.box.background = element_rect(fill = "#FCFCF5", color = "black", linewidth = 1),
    aspect.ratio = 1
  )

ggsave(
  "figures/train_metrics_plt.pdf", metrics_plt,
  width = 5, height = 4, scale = 2.5,
  device = cairo_pdf, antialias = "subpixel"
)
