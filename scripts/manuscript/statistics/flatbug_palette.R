fb_colors <- c(
  "lousegray" = "#2f4746",
  "collembolight" = "#c9d7d2",
  "eyeflyred" = "#c20404",
  "hoveryellow" = "#e3942f",
  "expurple" = "#a765bd",
  "beetleblue" = "#33387c",
  "elytragreen" = "#44ac4c",
  "waspcoffee" = "#251003",
  "mothbeige" = "#ede9c7"
)

fb_palette_main <- fb_colors
fb_palette_main_text <- c(
  "white" = "#ffffff",
  "black" = "#000000",
  "white" = "#ffffff",
  "black" = "#000000",
  "black" = "#000000",
  "white" = "#ffffff",
  "black" = "#000000",
  "white" = "#ffffff",
  "black" = "#000000"
)

fb_palette_simple <- fb_colors[c("beetleblue", "eyeflyred", "elytragreen", "expurple", "hoveryellow")]
fb_palette_contrast <- fb_colors[c("waspcoffee", "lousegray", "mothbeige")]
fb_palette_RdBu <- fb_colors[c("eyeflyred", "beetleblue")]
fb_palette_RdLi <- fb_colors[c("eyeflyred", "collembolight")]
fb_palette_RdWiBu <- c(
  fb_colors["eyeflyred"],
  "white" = "#ffffff",
  fb_colors["beetleblue"]
)
fb_palette_RdYl <- fb_colors[c("eyeflyred", "hoveryellow")]
fb_palette_Brown <- fb_colors[c("waspcoffee", "mothbeige")]

fb_palettes <- list(
  main = fb_palette_main,
  simple = fb_palette_simple,
  contrast = fb_palette_contrast,
  RdBu = fb_palette_RdBu,
  RdLi = fb_palette_RdLi,
  RdWiBu = fb_palette_RdWiBu,
  RdYl = fb_palette_RdYl,
  Brown = fb_palette_Brown,
  main_text = fb_palette_main_text
)

fb_d_palettes <- names(fb_palettes)[c(1:5, 9)]
fb_c_palettes <- names(fb_palettes)[3:8]

fb_gen_d <- function(palette = "main", direction = 1, lighten = 0) {
  pal_name <- match.arg(palette, fb_d_palettes)
  pal <- fb_palettes[[pal_name]]
  if (direction == -1) {
    pal <- rev(pal)
  }
  if (lighten != 0) {
    pal <- colorspace::lighten(pal, lighten)
  }
  function(n) {
    if (n > length(pal)) {
      stop(paste0("Palette ", pal_name, " has only ", length(pal), " colors"))
    }
    unname(pal[1:n])
  }
}

fb_gen_c <- function(palette = "Brown", direction = 1, ...) {
  pal_name <- match.arg(palette, fb_c_palettes)
  pal <- fb_palettes[[pal_name]]
  colorRampPalette(pal, ...)
}

scale_fill_flatbug <- function(palette = "main", direction = 1, lighten = 0, ...) {
  ggplot2::discrete_scale(
    "fill",
    palette = fb_gen_d(palette, direction, lighten),
    ...
  )
}

scale_color_flatbug <- function(palette = "main", direction = 1, lighten = 0, ...) {
  ggplot2::discrete_scale(
    "color",
    palette = fb_gen_d(palette, direction, lighten),
    ...
  )
}

scale_fill_flatbug_c <- function(palette = "Brown", direction = 1, ...) {
  pal <- fb_gen_c(palette = palette, direction = direction)

  ggplot2::scale_fill_gradientn(colors = pal(256), ...)
}

scale_color_flatbug_c <- function(palette = "Brown", direction = 1, ...) {
  pal <- fb_gen_c(palette = palette, direction = direction)

  ggplot2::scale_color_gradientn(colors = pal(256), ...)
}

test_fb_palettes_colorblindness <- function() {
  all_plts <- tibble(
    type = c("d", "c"),
    palette = list(fb_d_palettes, fb_c_palettes)
  ) %>%
    unnest(palette) %>%
    mutate(
      plt = map2(palette, type, function(name, t) {
        if (t == "d") {
          colors <- fb_palettes[[name]]
          cnames <- names(colors)
          tibble(
            cname = factor(cnames, unique(cname))
          ) %>%
            ggplot(aes(1, 0, fill = cname)) +
            geom_tile(width = 0.9, height = 0.9, color = "black", linewidth = 1) +
            # geom_text(
            #   aes(y = 0.55, label = cname),
            #   color = "black",
            #   size = 5,
            #   fontface = "bold"
            # ) +
            scale_fill_flatbug(palette = name, guide = "none") +
            scale_y_continuous(expand = expansion(0, c(0, 0.05))) +
            scale_x_continuous(expand = expansion(0, 0.05)) +
            theme_void() +
            theme(
              plot.tag.position = "left",
              strip.text = element_text(face = "bold"),
              aspect.ratio = 1
            ) +
            labs(tag = name) +
            facet_wrap(~cname, scales = "free", nrow = 1)
        } else {
          tibble(
            x = 1:200,
            z = 1:200
          ) %>%
            ggplot(aes(x, 1, fill = z)) +
            geom_tile(aes(fill = x), width = 1, height = 1) +
            scale_fill_flatbug_c(palette = name, guide = "none") +
            theme_void() +
            theme(
              plot.tag.position = "left",
              strip.text = element_text(face = "bold")
            ) +
            labs(tag = name)
        }
      })
    )

  discrete_plt_data <- all_plts %>%
    filter(type == "d")

  discrete_plts <- wrap_plots(
    discrete_plt_data$plt,
    ncol = 1,
    heights = 1
  )
  print(
    discrete_plts +
      plot_annotation(
        title = "Discrete palettes"
      )
  )
  print(
    colorblindr::cvd_grid(
      discrete_plts &
        theme(
          strip.text = element_blank()
        )
    ) +
      plot_annotation(
        title = "Discrete palettes"
      )
  )

  continuous_plts <- all_plts %>%
    filter(type == "c") %>%
    pull(plt) %>%
    wrap_plots(ncol = 1, heights = 1)

  print(
    continuous_plts +
      plot_annotation(title = "Continuous palettes")
  )
  print(
    colorblindr::cvd_grid(continuous_plts) +
      plot_annotation(title = "Continuous palettes")
  )

  return(invisible())
}
