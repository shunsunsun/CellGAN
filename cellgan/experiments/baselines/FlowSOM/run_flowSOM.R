library(flowCore)
library(FlowSOM)

DEFAULT_MARKERS = c("CD3", "CD45", "CD4", "CD20", "CD33", "CD123", "CD14", "IgM", "HLA-DR", "CD7")
DEFAULT_SUB_LIMIT = 30
 
# # If running from RStudio, use this.
# current_path <- rstudioapi::getActiveDocumentContext()$path
# setwd(dirname(current_path))

args <- commandArgs(trailingOnly = TRUE)
dir.name <- getwd()
source(file.path(dir.name, "cellgan/experiments/baselines/FlowSOM/flowSOM_utils.R"))

###################################################
######## Data Loading and Preprocessing ###########
###################################################

inhibitor = toString(args[1])
strength = toString(args[2])
nruns = as.numeric(args[3])
DATA_DIR <- file.path(dir.name, "data", inhibitor)
RESULT_DIR <- file.path(dir.name, "results/baselines/FlowSOM", inhibitor, strength)
dir.create(RESULT_DIR, showWarnings = FALSE)

files_to_process <- list.files(DATA_DIR, pattern=strength)

print(paste0("Starting to load and process the .fcs files"))
num_celltypes = 0

training_data <- matrix(nrow=0, ncol= length(DEFAULT_MARKERS))
training_labels <- vector(mode = "numeric", length = 0L)

for (file in files_to_process){
  
  # Loading and read the data
  datafile <- file.path(DATA_DIR, file)
  data <- read_fcs_data(datafile)
  
  # Get only the relevant markers
  marker_choices_indices <- extract_marker_indices(DEFAULT_MARKERS, colnames(data))
  data <- data[, marker_choices_indices]
  
  # Check if the 
  num_cells_measured <- dim(data)[1]
  if (num_cells_measured >= DEFAULT_SUB_LIMIT){
    
    # Apply some data transformation
    data <- f_trans(data)
    training_data <- rbind(training_data, data)
    training_labels <- c(training_labels, rep(num_celltypes, num_cells_measured))
    #print(length(training_labels))
    
    num_celltypes <- num_celltypes + 1
    
    print(paste0("File: ", file, " loaded and processed."))
    print(paste0("File: ", file, " has ", toString(num_cells_measured), " cells."))
  
  }
  
}

training_data <- flowCore::flowFrame(training_data)

#######################################################################
###### Running FlowSOM - manual and automatic number of clusters ######
#######################################################################

for (i in seq(nruns)){

  # Automatic number of clusters
  fSOM_auto.res <- FlowSOM::ReadInput(training_data, transform = FALSE, scale = FALSE)
  fSOM_auto.res <- FlowSOM::BuildSOM(fSOM_auto.res, colsToUse = NULL)
  fSOM_auto.res <- FlowSOM::BuildMST(fSOM_auto.res)
  
  # Manual number of clusters
  # fSOM_man.res <- FlowSOM::ReadInput(training_data, transform = FALSE, scale = FALSE)
  # fSOM_man.res <- FlowSOM::BuildSOM(fSOM_man.res, colsToUse = NULL, xdim = 20, ydim = 20)
  # fSOM_man.res <- FlowSOM::BuildMST(fSOM_man.res)
  
  # Automatic meta clustering
  meta_auto <- FlowSOM::MetaClustering(fSOM_auto.res$map$codes, method = "metaClustering_consensus")
  clusters_auto <- meta_auto[fSOM_auto.res$map$mapping[, 1]]
  
  labels_file <- paste0("Act_labels_run_", i, ".csv")
  cluster_file <- paste0("FlowSOM_clusters_run_", i,".csv")
  
  write.csv(clusters_auto, file=file.path(RESULT_DIR, cluster_file), row.names = FALSE)
  write.csv(training_labels, file=file.path(RESULT_DIR, labels_file), row.names = FALSE)

}