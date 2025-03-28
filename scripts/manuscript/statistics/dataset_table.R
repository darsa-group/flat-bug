source("helpers/flatbug_init.R")
library(googlesheets4)
library(kableExtra)

reference_path <- "C:/Users/asger/OneDrive - Aarhus universitet/Documents/PhD/Projekter/Universal Arthropod Localization/references.bib"

parse_field <- function(line) {
  key_content <- str_split(line, " = ", 2) %>% 
    unlist
  if (length(key_content) != 2) stop(str_c("Invalid field: ", line))
  key <- key_content[1]
  content <- key_content[2] %>% 
    str_remove_all("^\\s*\\{|\\}\\s*,*\\s*$") %>% 
    str_replace_all("\\\\\\\\", "\\")
  names(content) <- key
  content
}

parse_fields <- function(text) {
  lines <- str_split_1(text, "\n") %>% 
    str_squish()
  lines <- lines[nchar(lines) > 3]
  do.call(c, lapply(lines, parse_field)) %>% 
    as.list
}

split_references <- function(content) {
  references <- content %>% 
    str_extract_all("@\\w+\\{[^@]+\\}") %>% 
    unlist
  
  keys <- references %>% 
    str_extract("@\\w+\\{([^,\\s]+)", group=T)
  
  str_split(references, "\n") %>% 
    lapply(function(x) str_c(x[2:(length(x) - 1)], collapse="\n")) %>% 
    lapply(parse_fields) %>% 
    set_names(keys)
}

set_indent <- function(lines) {
  if (is.character(lines) & any(str_detect(lines, "\n"))) {
    if (length(lines) != 1) stop("Invalid format.")
    lines <- str_split_1(lines, "\n")
  }
  if (length(lines) == 1) return(lines)
  if (any(str_detect(lines, "\\\\begin\\{"))) {
    indent <- rep(0, length(lines))
    indent[which(str_detect(lines, "\\\\begin\\{")) + 1] <- 1
    indent[which(str_detect(lines, "\\\\end\\{"))] <- -1
    indent <- cumsum(indent)
    indent <- sapply(indent, function(x) strrep(" ", x*4))
    lines <- str_c(indent, lines)
  }
  return(str_c(lines, collapse = "\n"))
}

reference_library_data <- read_file(reference_path) %>% 
  split_references()

doi_search <- function(doi, library) {
  if (is.na(doi) | nchar(doi) < 5) return(vector("character"))
  doi_pattern <- str_escape(doi)
  library_match <- sapply(library, function(reference) {
    if (any(sapply(reference, function(field) str_detect(field, doi_pattern)))) return(TRUE)
    return(FALSE)
  })
  if (!any(library_match)) return(NA_character_)
  return(names(library)[library_match])
}

googlesheets4::gs4_auth("asgersvenning@gmail.com")
id <- "19cvXEf9KETfwT4bYQ9FpGuEOlerAfPwvh1KcQ4P9JhI"
dataset_table <- read_sheet(id) %>% 
  select(!all_of(colnames(.)[str_detect(colnames(.), "^[\\.]{3}")])) %>% 
  select(!all_of(colnames(.)[str_detect(colnames(.), "^_")])) %>%
  select(where(~!all(is.na(.))))

dataset_table_cleaned <- dataset_table %>%
  rename(
    "subdataset"   = "sub_dataset",
    "images"       = "n_images",
  ) %>% 
  relocate(short_name, .before = 0) %>% 
  mutate(
    short_name = str_c("\\texttt{", short_name, "}"),
    instance_number = instance_number %>% 
      str_remove(" insect[s]*"),
    crowding = crowding %>% 
      str_remove(" crowding") %>% 
      str_replace("no", "none"),
    sensor = sensor %>% 
      str_replace("non-", "not "),
    across(contains("DOI_"), ~inset(.x, is.na(.x) | str_detect(.x, regex("NA", T)), "-"))
  )

dataset_table_cleaned %>% 
  mutate(
    citation_keys = across(contains("DOI")) %>% 
      apply(1, function(x) lapply(x, function(y) doi_search(y, reference_library_data))) %>% 
      lapply(unlist) %>% 
      lapply(as.vector)
  ) %>% 
  select(!contains("DOI")) %>% 
  mutate(
    subdataset = map2_chr(subdataset, citation_keys, function(n, k) {
      if (length(k) == 0) return(n)
      k <- k[!is.na(k)]
      if (length(k) == 0) return(n)
      cite_str <- str_c("\\shortcite{", str_c(k, collapse=", "), "}")
      return(str_c(n, cite_str))
    }),
    standardized = str_detect(sensor, "(?<!not\\s{0,3})standardi[zs]ed") %>% 
      ifelse(
        "\\cmark",
        "\\xmark"
      ),
    sensor = str_remove(sensor, ",\\s*(not)*\\s*standardi[zs]ed"),
    sensor = str_c(sensor, "\\hfill(", standardized, ")")
  ) %>% 
  select(!c(standardized, citation_keys)) %>% 
  kable(
    "latex",
    caption = "Dataset description and characteristics. The three-letter abbreviation and full names of each subdataset can be found in the first two columns, while image charactistics and content descriptions can be found in the remaining six columns.",
    col.names = c(
      "",
      "Name",
      "Count",
      "Context",
      "Taxonomic Coverage",
      "Instance count",
      "Crowding",
      "Sensor (Standardized)"
    ),
    align = "clrlllll",
    booktabs = T,
    longtable = F,
    escape = F
  ) %>% 
  kable_paper() %>% 
  add_header_above(
    c(" " = 2, "Images" = 6)
  ) %>% 
  row_spec(
    0,
    align = "c"
  ) %>% 
  column_spec(
    1,
    width = "0.75cm",
    bold = T,
    border_right = T
  ) %>% 
  column_spec(
    2,
    width = "4cm"
  ) %>% 
  column_spec(
    3,
    width = "1cm"
  ) %>% 
  column_spec(
    c(4, 5),
    width = "4.5cm"
  ) %>% 
  column_spec(
    6,
    width = "2.75cm"
  ) %>% 
  column_spec(
    7,
    width = "3.25cm"
  ) %>% 
  column_spec(
    8,
    width = "3.5cm"
  ) %>% 
  row_spec(seq(1, 100, 2), background = "gray!15") %>% 
  landscape %>% 
  as.character %>% 
  str_replace("(?<=\\\\begin\\{landscape\\})", "\n") %>% 
  str_split_1("\\n") %>% 
  {
    c(
      .[1:3],
      "\\setlength{\\tabcolsep}{3pt}",
      .[4:5],
      "\\label{tab:dataset}",
      .[6:length(.)]
    )
  } %>% 
  str_c(collapse = "\n") %>% 
  set_indent() %>% 
  cat

dataset_table_cleaned %>% 
  select(short_name, contains("DOI")) %>% 
  mutate(
    across(contains("DOI"), ~map_chr(., function(doi) {
      if (nchar(doi) < 5) return(doi)
      ckeys <- doi_search(doi, reference_library_data)
      doi_link <- str_c("\\href{https://www.doi.org/", doi, "}{", doi, "}")
      if (length(ckeys) == 0) return(doi_link)
      ckeys <- ckeys[!is.na(ckeys)]
      if (length(ckeys) == 0) return(doi_link)
      cite_str <- str_c("\\shortcite{", str_c(ckeys, collapse=", "), "}")
      return(str_c(doi_link, cite_str))
    }))
  ) %>% 
  kable(
    "latex",
    caption = "Subdataset data, reference and source",
    col.names = c(
      "",
      
      "Data",
      "Article",
      "Data Source"
    ),
    align = "rlccc",
    booktabs = T,
    escape = F
  ) %>% 
  kable_paper() %>% 
  add_header_above(
    c(" " = 1, "DOI" = 3)
  ) %>% 
  row_spec(
    0,
    align = "c"
  ) %>% 
  column_spec(
    1,
    width = "1cm",
    bold = T,
    border_right = T
  ) %>% 
  row_spec(seq(1, 100, 2), background = "gray!15") %>% 
  str_split_1("\\n") %>% 
  {
  c(
    .[1:3],
    "\\label{tab:subdataset_doi}",
    .[4:length(.)],
    ""
  )
  } %>% 
  set_indent() %>% 
  cat

