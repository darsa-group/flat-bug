library("data.table")
library("ggplot2")
# library("tidyverse")
# library(tibble)
# library(dplyr)
library("scales")
library("optparse")
 

 # Parse command line arguments - I am using this style so it is basically looks like argparse in Python
option_list = list(
	make_option(c("-i", "--input_directory"), type="character", default=NULL, 
			  help="input directory [default= %default]", metavar="character"),
    make_option(c("-o", "--output_directory"), type="character", default=".", 
              help="output file name [default= %default]", metavar="character")
); 
 
opt_parser = OptionParser(option_list=option_list)
opts = parse_args(opt_parser)
for (i in c("input_directory", "output_directory")) {
  if (is.null(opts[[i]])) {
	stop("Argument ", i, " is required")
  }
  # Remove trailing slash
  opts[[i]] = gsub("/$", "", opts[[i]])
  # Check if the directory exists
  if (!dir.exists(opts[[i]])) {
	stop("`", i, "` Directory ", opts[[i]], " does not exist")
  }
}

OUR_THEME <- theme_bw()

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}


RES_DIR = opts$input_directory # where all the CSV files are
files = list.files(RES_DIR, pattern="*.csv", full.names=TRUE)

all_data = lapply(files, function(f){
    dt = fread(f)
    dt[, filename := basename (f) ]
    dt[, contour_1:=NULL ]
    dt[, contour_2:=NULL ]
    dt
})

dt = rbindlist(all_data)
dt[, in_gt := idx_1 != -1]
dt[, in_im := idx_2 != -1]
dt[, dataset := sapply(strsplit(dt[, filename], "_"), function(e){e[[1]][1]})]
dt[, area := ifelse(contourArea_1 >0 , contourArea_1, contourArea_2)]

results <- dt[, .(precision = sum(in_gt & in_im)/ sum(in_im),
				recall = sum(in_gt & in_im)/ sum(in_gt),
				n_instances = .N)
				,
				by="dataset"
			]

write.csv2(results, paste0(opts$output_directory, "/results.csv"))

# print(as_tibble(dt))

binomial_mean_sdl <- function(x, na.rm = TRUE, conf.int=0.95, ...) {
	if (na.rm) x <- x[!is.na(x)]
	n <- length(x)
	m <- mean(x, na.rm = na.rm)
	ms <- sapply(1:200, function(...) mean(sample(x, n, replace = TRUE)))
	tail_prob <- (1 - conf.int) / 2
	ci <- as.vector(quantile(ms, c(tail_prob, 1 - tail_prob)))
	out <- c(ymin = ci[1], y = m, ymax = ci[2])
	return(out)
}


x_limits = c(30, 2e7)
y_hist_limits = c(0, 1200)

layers = function(y_name){list(
	stat_summary_bin(fun = mean, geom="point", size=0.3, color="firebrick"),
	stat_summary_bin(fun = mean, geom="line", linewidth=0.2, color="firebrick"),
	stat_summary_bin(fun.data = binomial_mean_sdl, geom="ribbon", fill="firebrick", alpha=0.2),
	geom_smooth(method=mgcv::bam, method.args = list(family = "binomial", discrete=TRUE), formula = y ~ s(x, bs="cs"), linewidth=0.5, color="royalblue"),
	geom_point(alpha=.1, shape='|'),
	scale_x_log10(name = 'Insect area (px)',
	              labels =  scientific_10,
				  expand = expansion()),
	scale_y_continuous(name=y_name, labels=scales::label_percent(accuracy=1), limits=c(-0.05,1.05)),
	OUR_THEME
)}
hist_layers = function(){list(
		geom_histogram(),
		scale_x_log10(name = NULL, labels = NULL) ,
		scale_y_continuous(limits = y_hist_limits, breaks = c(0,500,1000)),
		OUR_THEME
)}

p <- ggplot(dt[in_gt==T], aes(area, as.numeric(in_gt & in_im))) + 
	layers(y_name='Recall') + 
	facet_wrap(~dataset) +
	coord_cartesian(expand = FALSE)

ggsave(paste0(opts$output_directory, "/recall.png"), p, width=10, height=10)

p <- ggplot(dt[in_im==T], aes(area, as.numeric(in_gt & in_im))) + 
	layers(y_name='Precision') + 
	facet_wrap(~dataset) +
	coord_cartesian(expand = FALSE)

ggsave(paste0(opts$output_directory, "/precision.png"), p, width=10, height=10)
