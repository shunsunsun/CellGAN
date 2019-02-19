library(flowCore)
library(MLmetrics)

# Helper function file accompanying run_flowSOM.R

f_trans <- function(x, cofactor=5){
  modified = asinh(x/cofactor)
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

# compute_f_measure <- function(y_true, y_pred){
#   
#   y_true_unique <- unique(y_true)
#   y_pred_unique <- unique(y_pred)
#   
#   N <- length(y_true)
#   f_measure_i <- c()
#   
#   for (i in 1:length(y_true_unique)){
#     y_i <- y_true_unique[i]
#     f_measure_j <- c()
#     
#     temp_ind_y = which(y_true == y_i)
#     
#     binary_y_i <- rep(0, N)
#     binary_y_i[temp_ind_y] <- 1
#     
#     n_c_i <- length(temp_ind_y)
#     
#     for (j in 1:length(y_pred_unique)){
#       y_j <- y_pred_unique[j]
#       temp_ind_y_j <- which(y_pred == y_j)
#       
#       binary_y_j <- rep(0, N)
#       binary_y_j[temp_ind_y_j] <- 1
#       
#       f1_score = F1_Score(binary_y_i, binary_y_j)
#       f_measure_j <- c(f_measure_j, f1_score)
#       
#     }
#     
#     score <- (n_c_i/N) * max(f_measure_j)
#     f_measure_i <- c(f_measure_i, score)
#     
#   }
#   
#   return (sum(f_measure_i))
#   
# }

# compute_f_measure_uniformly_weighted <- function(y_true, y_pred){
#   
#   y_true_unique <- unique(y_true)
#   y_pred_unique <- unique(y_pred)
#   
#   N <- length(y_true)
#   f_measure_i <- c()
#   n_c_i_total <- c()
#   
#   rare_ones <- c(3, 4, 7, 9, 10, 13)
#   
#   for (i in 1:length(y_true_unique)){
#     y_i <- y_true_unique[i]
#     f_measure_j <- c()
#     
#     temp_ind_y = which(y_true == y_i)
#     
#     binary_y_i <- rep(0, N)
#     binary_y_i[temp_ind_y] <- 1
#     
#     n_c_i <- length(temp_ind_y)
#     
#     for (j in 1:length(y_pred_unique)){
#       y_j <- y_pred_unique[j]
#       temp_ind_y_j <- which(y_pred == y_j)
#       
#       binary_y_j <- rep(0, N)
#       binary_y_j[temp_ind_y_j] <- 1
#       
#       precision = Precision(binary_y_i, binary_y_j, positive = "1")
#       recall = Recall(binary_y_i, binary_y_j, positive = "1")
#       
#       f1_score = 2 * precision * recall / (precision + recall + 1e-8)
#       
#       f_measure_j <- c(f_measure_j, f1_score)
#       
#     }
#     
#     score <- max(f_measure_j)
#     f_measure_i <- c(f_measure_i, score)
#     n_c_i_total <- c(n_c_i_total, n_c_i)
#     
#   }
#   
#   weights_i <- rep(1, length(n_c_i_total))/length(n_c_i_total)
#   
#   return (sum(f_measure_i * weights_i))
#   
# }
# 


