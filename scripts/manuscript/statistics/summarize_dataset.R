fb_yolo_path <- "//wsl.localhost/Ubuntu/home/asger/data/flat-bug/fb_yolo"

fb_yolo_all_files <- list.files(fb_yolo_path, recursive = T)

fb_yolo_clean <- tibble(
  path = fb_yolo_all_files
) %>% 
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

# Count images per dataset
fb_yolo_clean %>% 
  filter(!str_detect(path, "00-prospective")) %>%
  filter(split != "metadata" & type == "images") %>%
  count(split) %>% 
  pivot_wider(names_from = split, values_from = n) %>% 
  mutate(total = train + val)

# Count instances
library(furrr)
plan("multisession", workers = future::availableCores() - 2)

fb_yolo_label_instances <- fb_yolo_clean %>% 
  filter(type == "labels" & ext == "txt") 
  