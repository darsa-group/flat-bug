theme_set(
  ggpubr::theme_pubr(
    base_family = "CMU Serif",
    legend = "right"
  ) +
    theme(
      strip.text = element_text(hjust = 0.5, size = 16, face = "bold"),
      strip.text.y.right = element_text(angle = 0),
      strip.text.y.left = element_text(angle = 0),
      title = element_text(hjust = 0.5, size = 16, face = "bold"),
      plot.title = element_text(hjust = 0.5, size = 20, face = "plain"),
      legend.title = element_text(hjust = 0.5),
      legend.text = element_text(size = 14, hjust = 1)
      # panel.border = element_rect(fill = NA, color = "black", linewidth = 1)
    )
)

source("flatbug_palette.R")
