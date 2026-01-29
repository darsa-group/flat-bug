source("helpers/flatbug_init.R")
library(googlesheets4)

dataset_table <- googlesheets4::gs4_auth("asgersvenning@gmail.com")
id <- "19cvXEf9KETfwT4bYQ9FpGuEOlerAfPwvh1KcQ4P9JhI"

dataset_simple_characteristics <- read_sheet(id) %>% 
  select(!all_of(colnames(.)[str_detect(colnames(.), "^[\\.]{3}")])) %>% 
  select(!all_of(colnames(.)[str_detect(colnames(.), "^_")])) %>%
  select(where(~!all(is.na(.)))) %>% 
  select(short_name, context, instance_number, crowding, sensor) %>% 
  transmute(
    short_name,
    standardized = if_else(
      str_detect(sensor, "(?<!no[nt][-\\s]{0,1})[Ss]tandardi[sz]ed\\s*$"),
      "yes",
      "no"
    ) %>% 
      factor(c("yes", "no")),
    sensor = str_remove_all(sensor, ",.+$|\\s*\\(\\w+\\)\\s*") %>% 
      factor(c(
        "flatbed scanner",
        "camera",
        "microscope"
      )),
    context = case_when(
      str_detect(context, "(\\s|^)[lL]ive\\s") ~ "live individuals",
      str_detect(context, "ethanol|liquid") ~ "ethanol/liquid",
      str_detect(context, "sticky") ~ "sticky card"
    ) %>% 
      factor(c("live individuals", "sticky card", "ethanol/liquid")),
    crowding = crowding %>% 
      factor(
        c(
          "no crowding",
          "often adjacent", 
          "rarely touching", 
          "sometimes touching", 
          "often touching", 
          "very often touching", 
          "often overlapping", 
          "occasional crowding"
        )
      ) %>% 
      fct_collapse(
        `sometimes touching` = c(
          "rarely touching",
          "sometimes touching",
          "often touching"
        ),
        `very often touching or overlapping` = c(
          "very often touching",
          "often overlapping"
        ) 
      ),
    `instance count` = instance_number %>% 
      str_remove("\\s*insects?\\s*") %>% 
      factor(
        c(
          "single",
          "one or few",
          "few",
          "several",
          "multiple",
          "high density",
          "very high density"
        )
      ) %>% 
      fct_collapse(
        `one or few` = c(
          "single",
          "one or few",
          "few"
        ),
        `multiple` = c(
          "several",
          "multiple"
        ),
        `very many` = c(
          "high density",
          "very high density"
        )
      )
  ) %>% 
  mutate(
    across(
      where(is.factor), 
      ~ .x %>%
        fct_relabel(str_to_sentence) %>% 
        fct_relabel(str_wrap, 20)
    )
  )

model_metrics_with_characteristics <- "data/compare_backbone_sizes_combined_recomputed.csv" %>% 
  read_csv(show_col_types = F) %>% 
  mutate(
    dataset = str_remove(dataset, "^01-partial-"),
    model_size = factor(model_size, levels = c("L", "M", "S", "N"))
  ) %>% 
  select(model_size, short, n, F1, F1_lower, F1_upper) %>% 
  left_join(
    dataset_simple_characteristics,
    by = c("short" = "short_name")
  ) 

characteristics <- c("standardized", "sensor", "context", "crowding", "instance count")
semantic_plt <- tibble(
  characteristic = characteristics
  ) %>% 
  mutate(
    data = map(characteristic, function(var) {
      model_metrics_with_characteristics %>% 
        select(!any_of(setdiff(characteristics, var))) %>% 
        rename(all_of(c(value = var)))
    }),
    plt = map2(characteristic, data, function(var, df) {
      p <- df %>% 
        drop_na %>% 
        ggplot(aes(value, F1)) +
        geom_boxplot(
          aes(fill=model_size),
          outliers = F,
          key_glyph = draw_key_point
        ) +
        ggbeeswarm::geom_quasirandom(
          aes(group=model_size),
          shape=16,size=1.5,dodge.width=0.75,width=0.075,
          color="black"
        ) +
        scale_fill_flatbug() +
        scale_y_continuous(labels = scales::label_percent()) +
        labs(x = NULL, y = NULL, title = str_to_sentence(var), fill = "Model\nsize")
      if (var %in% c("sensor", "context", "instance count")) {
        p <- p +
          theme(
            axis.line.y.left = element_blank(),
            axis.text.y = element_blank(),
            axis.ticks.y = element_blank()
          )
      }
      p
    })
  ) %>% 
  pull(plt) %>% 
  patchwork::wrap_plots(
    guides = "collect",
    design = "AAAABBBBCCCC\nDDDDDDDEEEEE"
  ) &
  guides(
    fill = guide_legend(
      override.aes = list(
        shape = 21,
        color = "black",
        stroke = 1,
        size = 7
      ),
      nrow = 2
    )
  ) &
  theme(
    legend.position = "bottom",
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    panel.grid.major.y = element_line(color = "gray75", linewidth = 0.25, linetype = "solid"),
    panel.grid.minor.y = element_line(color = "gray75", linewidth = 0.25, linetype = "dashed"),
  )

ggsave(
  "figures/semantic_metrics.pdf", semantic_plt,
  device = cairo_pdf, width = 8, height = 10,
  scale = 1.5
)
