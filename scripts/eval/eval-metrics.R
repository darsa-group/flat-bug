library("data.table", warn.conflicts=FALSE)
library("ggplot2", warn.conflicts=FALSE)
library("dplyr", warn.conflicts=FALSE)
# library("tibble")
# library(dplyr)
library("scales", warn.conflicts=FALSE)
library("optparse", warn.conflicts=FALSE)
library("magrittr", warn.conflicts=FALSE)

 # Parse command line arguments - I am using this style so it looks like argparse in Python
option_list <- list(
  make_option(c("-i", "--input_directory"), type = "character", default = NULL,
              help = "input directory [default= %default]",
              metavar = "character"),
  make_option(c("-o", "--output_directory"), type = "character", default = ".",
              help = "output file name [default= %default]",
              metavar = "character")
)

opt_parser <- OptionParser(option_list = option_list)
opts <- parse_args(opt_parser)
for (i in c("input_directory", "output_directory")) {
  if (is.null(opts[[i]])) {
    stop("Argument ", i, " is required")
  }
  # Remove trailing slash
  opts[[i]] <- gsub("/$", "", opts[[i]])
  # Check if the directory exists
  if (!dir.exists(opts[[i]])) {
    stop("`", i, "` Directory ", opts[[i]], " does not exist")
  }
}

OUR_THEME <- theme_bw()

scientific_10 <- function(x) {
  x %>%
    scales::scientific_format()() %>%
    stringr::str_replace("(?<=e)\\+?0?", "") %>%
    stringr::str_replace("1(?=e)", "") %>%
    stringr::str_replace("(?<=^\\d)", " %*% ") %>%
    stringr::str_replace("e", "10^") %>%
    str2expression
}


RES_DIR <- opts$input_directory # where all the CSV files are
files <- list.files(RES_DIR, pattern = "*.csv", full.names = TRUE)
files <- files[!grepl("combined_results", files)]

all_data <- lapply(files, function(f) {
  f %>%
    fread(sep = ";") %>%
    as_tibble %>%
    mutate(
      filename = basename(f),
      countour_1 = NULL,
      countour_2 = NULL
    )
})

dt <- do.call(bind_rows, all_data) %>%
  mutate(
    in_gt = idx_1 != -1,
    in_im = idx_2 != -1,
    dataset = purrr::map_chr(filename, ~ stringr::str_extract(.x, "^[^_]+")),
    area = ifelse(contourArea_1 > 0, contourArea_1, contourArea_2)
  )

results <- dt %>%
  group_by(dataset) %>%
  summarize(
    precision = sum(in_gt & in_im) / sum(in_im),
    recall = sum(in_gt & in_im) / sum(in_gt),
    n_instances = n()
  )

n_datasets <- length(unique(dt$dataset))

fwrite(results, paste0(opts$output_directory, "/results.csv"))

binomial_mean_qci <- function(x, na.rm = TRUE, conf.int = 0.95, ...) {
  if (na.rm) x <- x[!is.na(x)]
  n <- length(x)
  m <- mean(x, na.rm = na.rm)
  ms <- sapply(1:200, function(...) mean(sample(x, n, replace = TRUE)))
  tail_prob <- (1 - conf.int) / 2
  ci <- as.vector(quantile(ms, c(tail_prob, 1 - tail_prob)))
  out <- c(ymin = ci[1], y = m, ymax = ci[2])
  return(out)
}

x_limits <- c(30, 2e7)
y_hist_limits <- c(0, 1200)

layers <- function(y_name) {
  list(
    stat_summary_bin(
      fun = mean,
      geom = "point",
      size = 0.3,
      color = "firebrick"
    ),
    stat_summary_bin(
      fun = mean,
      geom = "line",
      linewidth = 0.2,
      color = "firebrick"
    ),
    stat_summary_bin(
      fun.data = binomial_mean_qci,
      geom = "ribbon",
      fill = "firebrick",
      alpha = 0.2
    ),
    geom_smooth(
      method = mgcv::bam,
      method.args = list(family = "binomial", discrete = TRUE),
      formula = y ~ s(x, bs = "cs"),
      linewidth = 0.5,
      color = "royalblue"
    ),
    geom_point(alpha = .1, shape = "|"),
    scale_x_log10(
      name = "Insect area (px)",
      labels =  scientific_10,
      expand = expansion()
    ),
    scale_y_continuous(
      name = y_name,
      labels = scales::label_percent(accuracy = 1),
      limits = c(-0.05, 1.05)
    ),
    OUR_THEME
  )
}
hist_layers <- function() {
  list(
    geom_histogram(),
    scale_x_log10(name = NULL, labels = NULL),
    scale_y_continuous(limits = y_hist_limits, breaks = c(0, 500, 1000)),
    OUR_THEME
  )
}
pdf(paste0(opts$output_directory, "/eval-plots.pdf"), w = 10, h = 10)

recall_plot <- dt %>%
  filter(in_gt) %>%
  ggplot(aes(area, as.numeric(in_gt & in_im))) +
  layers(y_name = "Recall") +
  coord_cartesian(expand = FALSE)

precision_plot <- dt %>%
  filter(in_im) %>%
  ggplot(aes(area, as.numeric(in_gt & in_im))) +
  layers(y_name = "Precision") +
  coord_cartesian(expand = FALSE)

if (n_datasets > 1) {
  recall_plot <- recall_plot + facet_wrap(~dataset)
  precision_plot <- precision_plot + facet_wrap(~dataset)
}

print(recall_plot)
print(precision_plot)

dummy = dev.off()
