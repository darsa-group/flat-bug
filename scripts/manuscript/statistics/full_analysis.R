#################### User Control #################### 

# Perform (very expensive) evaluation aggregation
do_recompute <- FALSE 

# Do not run any scripts 
dry_run      <- FALSE 

#######################################################
library(progressr)

handlers(global=T)
handlers(handler_pbcol(
  adjust = 1.0,
  complete = function(s) cli::bg_blue(cli::col_black(s)),
  incomplete = function(s) cli::bg_none(cli::col_white(s)),
))

execute_script <- function(file, execute=T) {
  if (execute) source(file) else Sys.sleep(0.5)
}

reset_dev <- function() {
  while (tryCatch({dev.off(); T}, error = function(...) F)) {}
}

main <- function() {
  pb <- progressr::progressor(steps = 8 + do_recompute, auto_finish = F)
  
  pb("Setting up environment...", amount=0)
  options(full_analysis_running = T)
  if (!dir.exists("data")) dir.create("data")
  if (!dir.exists("figures")) dir.create("figures")
  if (!dir.exists("tiles")) dir.create("tiles")
  execute_script("helpers/flatbug_init.R", !dry_run)
  if (!dry_run) make_data_file()
  reset_dev()
  pb()
  
  pb("Downloading raw evaluation results...", amount=0)
  execute_script("helpers/flatbug_download_results.R", !dry_run)
  pb()
  
  if (do_recompute) {
    pb("Recomputing statistics...", amount=0)
    execute_script("flatbug_recompute_statistics.R", !dry_run)  
    pb()
  }
  
  pb("Summarizing dataset statistics...", amount=0)
  execute_script("summarize_dataset.R", !dry_run)
  pb()
  
  pb("Analyzing experiment 1...", amount=0)
  execute_script("compare_backbone_sizes.R", !dry_run)
  pb()
  
  pb("Analyzing experiment 2...", amount=0)
  execute_script("leave_one_out.R", !dry_run)
  pb()
  
  pb("Analyzing experiment 3...", amount=0)
  execute_script("leave_two_out.R", !dry_run)
  pb()
  
  pb("Creating appendix figures (1/2)...", amount=0)
  execute_script("justify_cutoff32.R", !dry_run)
  pb()
  
  pb("Creating appendix figures (2/2)...", amount=0)
  execute_script("ap_curve.R", !dry_run)
  pb()
  
  reset_dev()
  options(full_analysis_running = F)
  Sys.sleep(0.5)
  
  invisible()
}

main()

