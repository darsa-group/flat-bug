conf_IoU_plt <- tibble(
  model = c("N", "S", "M", "L") %>% 
    factor(rev(.))
) %>% 
  mutate(
    data = map(model, function(m) {
      "data/compare_backbone_sizes_{model}_full.csv" %>%
        str_glue(model=m) %>% 
        data.table::fread() %>% 
        as_tibble %>% 
        filter(IoU > 0) 
    })
  ) %>% 
  unnest(data) %>% 
  mutate(
    model = fct_recode(model, Nano = "N", Small = "S", Medium = "M", Large = "L")
  ) %>% 
  ggplot(aes(conf2, IoU)) +
  geom_histogram(
    aes(x = conf2, y = after_stat(-ncount*0.175)),
    inherit.aes = F,
    breaks = seq(0, 1, 0.01),
    position = position_nudge(y = 0.199),
    fill = "gray85", color = "gray85"
  ) +
  geom_histogram(
    aes(y = conf2, x = after_stat(-ncount*0.175)),
    inherit.aes = F,
    breaks = seq(0, 1, 0.01),
    position = position_nudge(x = 0.199),
    fill = "gray85", color = "gray85"
  ) +
  geom_bin2d(
    aes(fill = after_stat(count), color = after_scale(fill)), 
    breaks = seq(0, 1, 1/65),
    linewidth = 0.15
  ) +
  geom_vline(xintercept = 0.2, linetype = "dashed", color = "royalblue") +
  annotate(
    x = 0.175, y = 0.21, label = "Confidence threshold",
    geom = "text", vjust = 0, hjust = 0, angle = 90,
    color = "royalblue"
  ) +
  geom_hline(yintercept = 0.2, linetype = "dashed", color = "forestgreen") +
  annotate(
    x = 0.21, y = 0.175, label = "IoU threshold",
    geom = "text", vjust = 1, hjust = 0,
    color = "forestgreen"
  ) +
  scale_fill_viridis_c(option = "A", trans = "log10", direction = -1, n.breaks = 10) +
  scale_x_continuous(limits = 0:1, expand = expansion(), labels = scales::label_percent(), n.breaks = 6) +
  scale_y_continuous(limits = 0:1, expand = expansion(), labels = scales::label_percent(), n.breaks = 6) +
  labs(
    x = "Confidence", 
    y = "IoU",
    fill = "Count"
  ) +
  facet_wrap(~model, nrow = 2) +
  theme(
    legend.key.height = unit(10, "lines"),
    legend.key.width = unit(2, "lines"),
    aspect.ratio = 1
  )

ggsave(
  "figures/conf_iou_relationship.pdf", conf_IoU_plt,
  width = 7, height = 6, scale = 2.25,
  device = cairo_pdf
  
)