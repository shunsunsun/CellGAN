library(flowCore)

# Helper function file accompanying run_flowSOM.R

f_trans <- function(x, cofactor=5){
  modified = asinh(1/(x * cofactor))
  return(modified)
}

get_marker_names <- function(x){
  
  split_x <- strsplit(x, "\\(")
  marker_name <- split_x[[1]][1]
  return(marker_name)
}

read_fcs_data <- function(filename){
  
  fcs_file <- flowCore::read.FCS(filename, transformation = FALSE)
  data <- exprs(fcs_file)
  data <- data[, 3:35]
  
  # Change marker names
  names <- colnames(data)
  marker_names <- sapply(X=names, FUN=get_marker_names)
  marker_names <- unname(marker_names)
  
  colnames(data) <- marker_names
  
  return(data)
}

extract_marker_indices <- function(markers_of_interest, marker_names){
  
  return(which(marker_names %in% markers_of_interest))
  
}

plot_marker_distributions <- function(){}

plot_pca <- function(){}



