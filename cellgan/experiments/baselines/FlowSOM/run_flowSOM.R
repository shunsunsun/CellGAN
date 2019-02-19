library(flowCore)
library(FlowSOM)

DEFAULT_MARKERS = c("CD3", "CD45", "CD4", "CD20", "CD33", "CD123", "CD14", "IgM", "HLA-DR", "CD7")
DEFAULT_SUB_LIMIT = 30

# If running from RStudio, use this.
current_path <- rstudioapi::getActiveDocumentContext()$path
setwd(dirname(current_path))

#If running from terminal use this:
#dir.name <- getSrcDirectory(function(x) {x})
#setwd(dir.name)
source("flowSOM_utils.R")

###################################################
######## Data Loading and Preprocessing ###########
###################################################

inhibitor = "AKTi"
strength = "A02"
DATA_DIR <- paste0("../../../../data/", inhibitor)

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
    
    num_celltypes <- num_celltypes + 1
    
    # Apply some data transformation
    data <- f_trans(data)
    training_data <- rbind(training_data, data)
    training_labels <- c(training_labels, rep(num_celltypes, num_cells_measured))
    
    print(paste0("File: ", file, " loaded and processed."))
    print(paste0("File: ", file, " has ", toString(num_cells_measured), " cells."))
  
  }
  
}

training_data <- flowCore::flowFrame(training_data)

#######################################################################
###### Running FlowSOM - manual and automatic number of clusters ######
#######################################################################

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

write.csv(clusters_auto, file="FlowSOM_clusters_A02.csv", row.names = FALSE)

# Manual meta clustering
#TODO(vsomnath): Set max to 20 as that was the number of experts we used?
# meta_man <- FlowSOM::MetaClustering(fSOM_man.res$map$codes, method = "metaClustering_consensus", max = 20)
# clusters_man <- meta_man[fSOM_man.res$map$mapping[, 1]]

#print("F-Measure using automatic clusters")
#print(compute_f_measure_uniformly_weighted(training_labels, clusters_auto))

#print("F-Measure using manual clusters")
#print(compute_f_measure_uniformly_weighted(training_labels, clusters_man))


