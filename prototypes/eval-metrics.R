library("data.table")
library("ggplot2")
library("scales")

RES_DIR = "/home/quentin/Desktop/flat-bug-val-res-premerge" # where all the CSV files are


OUR_THEME <- theme_bw()

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}


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

library("data.table")
library("ggplot2")
library("scales")

RES_DIR = "/home/quentin/Desktop/flat-bug-val-res-premerge" # where all the CSV files are


OUR_THEME <- theme_bw()

scientific_10 <- function(x) {
  parse(text=gsub("e", " %*% 10^", scales::scientific_format()(x)))
}


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

dt[, .(precision = sum(in_gt & in_im)/ sum(in_im),
		recall = sum(in_gt & in_im)/ sum(in_gt),
		n_instances = .N,
		n_files = length(unique(filename)))
		,
		by="dataset"
	]




x_limits = c(30, 2e7)
y_hist_limits = c(0,1200)

layers = function(y_name){list(
	geom_smooth(method="gam", method.args = list(family = "binomial")),
	geom_point(alpha=.1, shape='|'),
	scale_x_log10(name = 'Insect area (px)', limits=x_limits,
	              labels =  scientific_10),
	scale_y_continuous(name=y_name), OUR_THEME
)}
hist_layers = function(){list(
		geom_histogram(),
		scale_x_log10(name = NULL, labels = NULL, limits=x_limits) ,
		scale_y_continuous(limits = y_hist_limits, breaks = c(0,500,1000)),
		OUR_THEME
)}


p <- ggplot(dt[in_gt==T], aes(area, as.numeric(in_gt & in_im))) + layers(y_name='Recall') + facet_wrap(~dataset)
p
p <- ggplot(dt[in_im==T], aes(area, as.numeric(in_gt & in_im))) + layers(y_name='Precision') + facet_wrap(~dataset)
p
