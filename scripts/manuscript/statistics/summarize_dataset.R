fb_yolo_all_files <- read_lines(str_c(fb_repository, "/folder_index.txt")) %>% 
  magrittr::extract(str_detect(., "fb_yolo")) %>% 
  str_remove("^fb_yolo/")

fb_yolo_clean <- tibble(
  path = fb_yolo_all_files
) %>% 
  filter(!str_detect(path, "00-prospective")) %>%
  mutate(
    ext = str_extract(path, "(?<=\\.).{2,4}$") %>% 
      tolower %>% 
      factor,
    split = str_extract(path, "(?<=^insects/(images|labels)/)[^/]+") %>% 
      factor(c("train", "val")),
    type = str_extract(path, ("(?<=^insects/)[^/]+")) %>% 
      inset(is.na(.), "metadata") %>% 
      factor(c("metadata", "labels", "images")),
    dataset = str_extract(path, "(?<=^insects/(images|labels)/(train|val)/)[^/\\.]+") %>% 
      str_remove("^0[01]-(prospective|partial)-") %>% 
      str_extract("^[^_]+"),
    dataset = ifelse(dataset == "instances", NA_character_, dataset)
  )

# Count subdatasets
n_subdatasets <- fb_yolo_clean %>% 
  distinct(dataset) %>%
  drop_na %>% 
  nrow

# Count images per dataset
fb_image_split_count <- fb_yolo_clean %>% 
  filter(split != "metadata" & type == "images") %>%
  count(split) %>% 
  pivot_wider(names_from = split, values_from = n) %>% 
  mutate(total = train + val)

# Count instances
library(furrr)
plan(multisession, workers=max(1, future::availableCores() - 2))
fb_instance_split_count <- fb_yolo_clean %>% 
  filter(type == "labels" & ext == "txt")  %>%
  mutate(
    path = str_c(fb_repository, "/fb_yolo/", path)
  ) %>% 
  mutate(
    n = future_map_int(path, function(x) length(read_lines(URLencode(x))), .progress = show_progress(), .options = furrr_options(seed=NULL))
  ) %>% 
  summarize(
    n = sum(n),
    .by = "split"
  ) %>% 
  pivot_wider(names_from = split, values_from = n) %>% 
  mutate(total = train + val)
plan(sequential)

dataset_summary_latex <- "\\pgfkeys{
    /dataset/.cd,
    images/.initial={{total_images}},
    train_images/.initial={{train_images}},
    val_images/.initial={{val_images}},
    instances/.initial={{total_instances}},
    train_instances/.initial={{train_instances}},
    val_instances/.initial={{val_instances}},
    subdatasets/.initial={{n_subdatasets}}
}
\\NewDocumentCommand{\\dataset}{m}{%
    \\pgfkeysvalueof{/dataset/#1}%
}
" %>% 
  str_glue(
    total_images = fb_image_split_count$total,
    train_images = fb_image_split_count$train,
    val_images = fb_image_split_count$val,
    total_instances = fb_instance_split_count$total,
    train_instances = fb_instance_split_count$train,
    val_instances = fb_instance_split_count$val,
    .open = "{{", 
    .close = "}}"
  )

add_group("Dataset - Summary")
write_data("Dataset - Summary", dataset_summary_latex)

