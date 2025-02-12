leave_two_out_full <- "data/leave_two_out_combined_recomputed.csv" %>% 
  read_csv(show_col_types = F) %>% 
  mutate(
    left_short = short_name(left),
    right_short = short_name(right),
    short = short_name(dataset)
  )

leave_two_out_full_clean <- leave_two_out_full %>% 
  mutate(
    across(c(left, right, dataset), ~str_remove(.x, "^01-partial-"))
  ) %>% 
  add_count(left, name = "nn") %>% 
  arrange(nn) %>% 
  mutate(
    left = factor(left, unique(c(left, right))),
    left_short = factor(left_short, unique(c(left_short, right_short))),
    right = factor(right, levels(left)),
    right_short = factor(right_short, levels(left_short)),
    dataset = factor(dataset, unique(c(levels(left), sort(unique(dataset))))),
    short = factor(short, unique(c(levels(left_short), sort(unique(short))))),
  ) %>% 
  select(!nn) %>% 
  bind_rows(., .) %>% 
  mutate(
    ltri = row_number() <= n() / 2,
    lt = left,
    lts = left_short,
    left = if_else(ltri, left, right),
    left_short = if_else(ltri, left_short, right_short),
    right = if_else(ltri, right, lt),
    right_short = if_else(ltri, right_short, lts)
  ) %>% 
  filter(!(ltri & left == right)) %>% 
  select(!c(ltri, lt, lts)) %>%
  mutate(
    across(contains("short"), ~factor(.x, sort(unique(as.character(short)))))
  ) %>% 
  arrange(short, left_short, right_short) 

write_csv(leave_two_out_full_clean, "data/leave_two_out_combined_recomputed_clean.csv")
write_rds(leave_two_out_full_clean, "data/leave_two_out_combined_recomputed_clean.rds")

# Convert from long tidy format to data cube
convert_to_cube <- function(data, metric) {
  metric <- substitute(metric)
  data %>% 
    drop_na %>% 
    select(left_short, right_short, short, metric) %>% 
    nest(results = !short) %>% 
    mutate(
      slice = map(results, function(d) {
        d %>% 
          pivot_wider(id_cols = left_short, names_from = right_short, values_from = metric) %>% 
          column_to_rownames("left_short") %>% 
          as.matrix 
      }) %>% 
        set_names(short)
    ) %>% 
    select(short, slice) %>% 
    pull(slice) %>% 
    abind::abind(along = 0) 
}

leave_two_out_cube_F1 <- convert_to_cube(leave_two_out_full_clean, F1) 
leave_two_out_cube_P  <- convert_to_cube(leave_two_out_full_clean, Precision)
leave_two_out_cube_R  <- convert_to_cube(leave_two_out_full_clean, Recall)

# Save the control results (model trained on all subdatasets)
extract_control <- function(data, metric) {
  metric <- substitute(metric)
  
  data %>% 
    filter(is.na(left_short)) %>% 
    select(short, metric) %>% 
    arrange(short) %>% 
    mutate(values = set_names(!!metric, short)) %>% 
    pull(values)
}


leave_two_out_control_F1 <- extract_control(leave_two_out_full_clean, F1)
leave_two_out_control_P <- extract_control(leave_two_out_full_clean, Precision)
leave_two_out_control_R <- extract_control(leave_two_out_full_clean, Recall)

l2o_delta <- function(cube, control, what, normalize=T) {
  slice <- cube[what,,]
  ctrl <- control[what]
  delta <- slice - ctrl
  if (normalize) {
    delta <- delta/ifelse(delta > 0, 1 - ctrl, ctrl)
  }
  delta
}

# # EDA plots
# lapply(dimnames(leave_two_out_cube_F1)[[2]], function(x) l2o_delta(leave_two_out_cube_F1, leave_two_out_control_F1, x)) %>% 
#   abind::abind(along = 0) %>% 
#   {
#     dimnames(.)[[1]] <- dimnames(.)[[2]]
#     .
#   } %>% 
#   apply(1, identity, simplify = F) %>% 
#   {map2(., names(.), function(x, n) plot_matrix(x, limits = c(-1, 1), title = n, text = T))} %>% 
#   patchwork::wrap_plots(guides = "collect", axes = "collect", axis_titles = "collect")

# Normalize (see paper on \delta vs \Delta) and extract relevant row
# Matrix, M, contains for each row the difference in normalized F1 between case 
# ij and jj against ii, where i and j correspond to different datasets left out.
# Metrics are evaluated only on the dataset of the correspond *row* (i).
#
#       M_{ij} = \frac{2}{3}\delta_{F1}^{ij} + \frac{1}{3}\delta_{F1}^{jj} - \delta_{F1}^{ii}
#              =   \frac{2}{3}(\delta_{F1}^{ij} - \delta_{F1}^{ii}) 
#                + \frac{1}{3}(\delta_{F1}^{jj} - \delta_{F1}^{ii})
#
# The first term corresponds to the extra decrease in performance when omitting
# the second (j) on top of the first (i) dataset, while the second term 
# corresponds to the extra decrease in performance when leaving out only the 
# second dataset (j). We will also call this the one-way redundancy \rho^1.
# Later we will use the two-way redundance simply defined as the average over
# the two asymmetrical cases of \rho^1 with the same indices:
#
#       \rho^2 = \frac{\rho^1 + (\rho^1)^T}{2}
#
compute_focal <- function(cube, ctrl) {
  sapply(dimnames(cube)[[2]], function(x) {
    row <- l2o_delta(cube, ctrl, x, T)[x,]
    row - row[x]
  })
}

focal_matrix_F1 <- compute_focal(leave_two_out_cube_F1, leave_two_out_control_F1)
focal_matrix_P  <- compute_focal(leave_two_out_cube_P, leave_two_out_control_P)
focal_matrix_R  <- compute_focal(leave_two_out_cube_R, leave_two_out_control_R)


focal_subdatasets <- focal_matrix_F1 %>% 
  dimnames %>% 
  first

focal_subdatasets_latex <- tibble(
  name = c("which", "subdatasets"),
  value = c(
    str_c(focal_subdatasets, collapse = ", "),
    as.character(length(focal_subdatasets))
  )
) %>% 
  mutate(
    ltx = str_c("\\defexperiment{3}{", name, "}{", value, "}")
  ) %>% 
  pull(ltx) %>% 
  str_c(collapse = "\n")

add_group("Experiment 3 - Subdatasets")
write_data("Experiment 3 - Subdatasets", focal_subdatasets_latex)

library(tidygraph)
library(ggraph)

# twr = two-way redundancy
twr_dist <- focal_matrix_F1 %>%
  add(t(.)) %>%
  divide_by(2) %>% 
  as.dist

l2o_mat_plt <- twr_dist %>% 
  as.matrix %>% 
  plot_matrix() +
  labs(fill = "Two-Way Redundancy") +
  theme(
    legend.position = "bottom",
    legend.title = element_text(vjust = 1),
    legend.key.width = unit(5, "lines")
  )

ggsave(
  "figures/leave_two_out_matrix.pdf",
  l2o_mat_plt,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 3,
  antialias = "subpixel"
)

anno_pos <- max(abs(twr_dist))/4 * c(1, 0.5, 0, -0.5, -1) 

l2o_tree_plt <- twr_dist %>%
  hclust("average") %>% 
  tidygraph::as_tbl_graph() %>% 
  mutate(
    parent = map_dfs_back_int(node_is_root(), .f = function(node, path, parent, graph, ...) {
      parent
    }),
    height = ifelse(leaf, height[parent] - 0.005, height)
  ) %>% 
  ggraph::ggraph("dendrogram", height = height) +
  geom_hline(
    yintercept = 0, 
    color = "firebrick", 
    linetype = "dashed", 
    linewidth = 0.75
  ) +
  annotate(
    "label",
    x = Inf,
    y = anno_pos,
    label = c("Antagonistic", "←", "Redundant", "→", "Synergistic"),
    vjust = 0.3,
    hjust = 0.5,
    fontface = "bold",
    size = c(4, 8, 4, 8, 4),
    fill = "white",
    label.size = 0,
    family = c("CMU Serif", "", "CMU Serif", "", "CMU Serif")
  ) +
  ggraph::geom_edge_elbow(
    linewidth = 0.5, 
    color = "gray25"
  ) +
  ggimage::geom_image(
    aes(
      x = ifelse(leaf, x, NA_real_),
      y = y,
      image = file.path("tiles", str_c(label, ".jpg"))
    ),
    image_fun = function(x) image_circlecut(magick::image_sample(x, 200), 1, T, 5),
    size = 0.055,
    position = position_nudge(y = 0.014)
  ) +
  geom_label(
    aes(x, y, label = ifelse(leaf, label, NA_character_)), 
    size = 3.5,
    hjust = 0,
    label.padding = unit(0.05, "lines"), 
    label.size = 0,
    family = "Courier New",
    fontface = "bold"
  ) +
  geom_label(
    aes(x,y,label = ifelse(!leaf, as.character(round(height, 3)), NA_character_)),
    size = 3,
    hjust = 1,
    vjust = 0.5,
    label.size = 0,
    label.padding = unit(0.1, "lines"),
    position = position_nudge(y = -0.001, x = 0.32),
    family = "CMU Serif"
  ) +
  scale_y_continuous(
    breaks = seq(-1, 1, 0.05), 
    minor_breaks = seq(-1, 1, 0.025),
    trans = scales::transform_reverse(),
    # trans  = scales::trans_new(
    #   name = "DUMMY",
    #   transform = function(x) -sign(x) * sqrt(abs(x)),
    #   inverse = function(x) -sign(x) * (x ^ 2),
    #   breaks = scales::breaks_extended
    # )
  ) +
  coord_flip(clip = "off") +
  theme(
    # panel.grid.minor.x = element_line(linewidth = 0.25, color = "gray35", linetype = "dashed"),
    # panel.grid.major.x = element_line(linewidth = 0.25, color = "gray35", linetype = "solid"),
    axis.text.x = element_text(),
    axis.ticks.x = element_line(linewidth = 0.75),
    axis.line.x = element_line(linewidth = 0.75),
    axis.title.x = element_text(),
    plot.margin = margin(1, 1, 0.25, 1, "lines")
  ) +
  labs(y = expression(bold('Two-Way Redundancy ' (rho^"2"))))

ggsave(
  "figures/leave_two_out_tree.pdf", 
  l2o_tree_plt,
  device = cairo_pdf,
  width = 4, height = 2,
  scale = 3,
  antialias = "subpixel"
)

twr_pcoa_ape <- twr_dist %>%
  {(1 + .)/2} %>%
  ape::pcoa()

n_PCs <- min(which(twr_pcoa_ape$values$Relative_eig >= twr_pcoa_ape$values$Broken_stick))
# Visualize how number of PCs are chosen
viz_pc_n_plt <- tibble(
  eigen = twr_pcoa_ape$values$Relative_eig, 
  stick = twr_pcoa_ape$values$Broken_stick,
) %>% 
  mutate(
    n = row_number()
  ) %>% 
  ggplot(aes(eigen, stick, label = n)) +
  geom_point(shape = 21, stroke = 0.75, size = 2) +
  geom_text(position = position_nudge(y = -0.005)) +
  geom_abline(slope = 1, intercept = 0, color = "firebrick", linetype = "dashed") +
  labs(title = str_c("Selected ", n_PCs, " Principal Components")) +
  theme(aspect.ratio = 1)

twr_pcoa <- twr_pcoa_ape %>%
  extract2("vectors") %>%
  as.data.frame() %>% 
  set_colnames(str_c("Axis.", 1:ncol(.))) %>% 
  rownames_to_column("short") %>% 
  as_tibble 

twr_pcoa_A1_min <- min(twr_pcoa$Axis.1)
twr_pcoa_A1_ran <- diff(range(twr_pcoa$Axis.1))
twr_pcoa_A2_min <- min(twr_pcoa$Axis.2)
twr_pcoa_A2_ran <- diff(range(twr_pcoa$Axis.2))

tidy_twr_pcoa_mst <- twr_pcoa[,2:(2+1)] %>%
  dist %>% 
  as.matrix %>% 
  as_tbl_graph %>% 
  to_minimum_spanning_tree(weights = weight) %>% 
  extract2("mst") %>% 
  activate("edges") %>% 
  as_tibble %>% 
  select(!weight) %>% 
  mutate(
    from_c1 = twr_pcoa$Axis.1[from],
    from_c2 = twr_pcoa$Axis.2[from],
    to_c1 = twr_pcoa$Axis.1[to],
    to_c2 = twr_pcoa$Axis.2[to],
    # mst_d = sqrt((to_c1 - from_c1)^2 + (to_c2 - from_c2)^2),
    mst_d = as.matrix(twr_dist)[matrix(c(from, to), ncol=2)],
    across(c(from, to), ~twr_pcoa$short[.x])
  ) %>% 
  mutate(
    across(c(from_c1, to_c1), ~(.x - twr_pcoa_A1_min)/twr_pcoa_A1_ran),
    across(c(from_c2, to_c2), ~(.x - twr_pcoa_A2_min)/twr_pcoa_A2_ran)
  )

compute_text_angle <- function(x1, y1, x2, y2) {
  # Calculate the angle perpendicular to the line segment
  angle <- (atan2(y2 - y1, x2 - x1) * 180 / pi + 90) %% 360
  
  # Adjust angles to keep text right-side-up
  angle <- ifelse(angle > 180, angle - 180, angle)
  
  return(angle - 90)
}

l2o_proj_plt <- twr_pcoa %>% 
  mutate(
    image = file.path("tiles", str_c(short, ".jpg")),
    Axis.1 = (Axis.1 - twr_pcoa_A1_min)/twr_pcoa_A1_ran,
    Axis.2 = (Axis.2 - twr_pcoa_A2_min)/twr_pcoa_A2_ran
  ) %>%
  mutate(
    xld = close_neighbor_dir(Axis.1, Axis.2, 1, "x"),
    yld = close_neighbor_dir(Axis.1, Axis.2, 1, "y"),
    xcld = close_neighbor_dir(Axis.1, Axis.2, 0.015, "x"),
    ycld = close_neighbor_dir(Axis.1, Axis.2, 0.015, "y")
  ) %>% 
  ggplot(aes(Axis.1, Axis.2, label = short)) +
  geom_segment(
    data = tidy_twr_pcoa_mst,
    aes(
      x = from_c1, 
      y = from_c2, 
      xend = to_c1, 
      yend = to_c2
    ),
    inherit.aes = F
  ) +
  geom_text(
    data = tidy_twr_pcoa_mst,
    aes(
      x = (from_c1 + to_c1)/2, 
      y = (from_c2 + to_c2)/2, 
      angle = compute_text_angle(from_c1, from_c2, to_c1, to_c2), 
      label = round(mst_d, 2),
      vjust = ifelse(sqrt((from_c1 - to_c1)^2 + (from_c2 - to_c2)^2) > 0.075, -0.5, -2.75),
    ),
    inherit.aes = F,
    hjust = 0.5
  ) +
  ggimage::geom_image(
    aes(image = image), 
    image_fun = function(x) image_circlecut(magick::image_sample(x, 200), 1, T, 5), 
    size = 0.05
  ) +
  geom_label(
    aes(
      x = Axis.1 + 0.05 * xld * (ycld != 0),
      y = Axis.2 + 0.0425 * yld * (xcld != 0 | (xcld == 0 & ycld == 0)),
    ), 
    fontface = "bold", 
    label.size = 0, 
    vjust = 0.5, hjust = 0.5,
    label.padding = unit(0.5, "mm")
  ) +
  labs(x = "PC 1", y = "PC 2") +
  coord_equal(xlim = 0:1, ylim = 0:1) +
  theme(
    aspect.ratio = 1,
    axis.text = element_blank(),
    axis.ticks = element_blank(),
    axis.line = element_line(
      arrow = arrow(type = "closed", angle = 30, length = unit(0.5, "lines"))
    )
  )

ggsave(
  "figures/leave_two_out_projection.pdf", 
  l2o_proj_plt,
  device = cairo_pdf,
  width = 4, height = 4,
  scale = 3,
  antialias = "subpixel"
)

l2o_comb_plt <- (l2o_tree_plt | l2o_proj_plt) +
  plot_annotation(
    tag_levels = "A", tag_prefix = "", tag_suffix = ")"
  ) &
  theme(
    plot.tag = element_text(family = "CMU Serif", face = "bold", size = 20, vjust = 1), 
    plot.tag.position = "left",
    plot.tag.location = "margin"
  )

ggsave(
  "figures/leave_two_out_combined.pdf", 
  l2o_comb_plt,
  device = cairo_pdf,
  width = 8, height = 4,
  scale = 3,
  antialias = "subpixel"
)
