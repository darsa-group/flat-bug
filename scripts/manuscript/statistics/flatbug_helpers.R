erda_repository <- "https://anon.erda.au.dk/share_redirect/OzZk71Y5SS"

googlesheets4::gs4_auth(email = "asgersvenning@gmail.com")

short_index <- googlesheets4::range_read(
  "1X3Dlj3hyT990B-sZWk8EWL-kWMNzhqrCYp4vD7cIkfg",
  "Sheet1",
  range = "A:C"
) %>%
  select(!ID) %>%
  rename(dataset = name, short_name = three_letter_code)

short_name <- Vectorize(memoise::memoise(function(x) {
  left_match <- str_detect(short_index$dataset, regex(str_escape(x), ignore_case = T))
  right_match <- str_detect(x, regex(str_escape(short_index$dataset), ignore_case = T))
  any_match <- left_match | right_match
  which_match <- which(any_match)
  if (length(which_match) != 1) {
    return(NA_character_)
  }
  return(short_index$short_name[which_match])
}))

f1_from_rp <- function(recall, precision) {
  f1 <- 2 * recall * precision / (recall + precision)
  f1[recall == 0 | precision == 0] <- 0
  return(f1)
}