get_cliff_delta <- function(data) {
  # List of comparisons
  model_list <- list(c("C-DCNN", "DCNN"), c("C-DCNN", "DFNN"),c("C-DCNN", "LR"),
                     c("C-DCNN", "RF"), c("DCNN", "DFNN"),c("DCNN", "LR"), c("DCNN", "RF"),
                     c("DFNN", "LR"),c("DFNN", "RF"), c("LR", "RF"))
  # converting data from long format to wide format
  data_wide <- data %>%
    filter(Model %in% c("DCNN", "C-DCNN", "LR", "RF", "DFNN")) %>%
    group_by(Model) %>%
    mutate(X = row_number()) %>%   # create pairing/block index
    ungroup() %>%
    select(X, Model, MAE) %>%
    pivot_wider(
      names_from = Model,
      values_from = MAE
    )
  
  # Calculating cliff's delta for all comparisons
  results <- pairwise_cliff_delta(data_wide, model_list)
  
  return(results)
  
}

get_sig_val_bonferronni <- function(p, n) {
  # Helper function to add significance stars in tables.
  if (n == 0) return(0)
  if (p < 0.001/n) return(sprintf("p < %.4f", 0.001/n))
  else if (p < 0.01/n) return(sprintf("p < %.3f", 0.01/n))
  else if (p < 0.05/n) return(sprintf("p < %.3f", 0.05/n))
  else return(p)
}

extract_wilcox_table <- function(model, results) {
  data.frame(
    `Comparison` = gsub("_", " ", results$comparison),
    `Estimated Median Difference in MAE` = round(model[["estimate"]],3),
    `p_adjusted`=  sapply(model$p.adj, get_sig_val_bonferronni, n = 10),
    `Effect size` = round(results$estimate,2),
    check.names = FALSE
  )
}

plot_results <- function(data, title, limits, breaks) {
  plot <- ggboxplot(
    data,
    x = "Model",
    y = "MAE",
    fill = "grey85", 
    colour = "black",
    fill = "Model",                 
    palette = "jco",                
    alpha = 0.6,                  
    add = "jitter",
    add.params = list(alpha = 0.5, size = 1.5, jitter = 0.2)  # soft scatter
  ) +
    theme_pubr() +
    theme(
      legend.position = "none",     # remove redundant legend
      axis.text.x = element_text(angle = 20, hjust = 1, size = 12),
      axis.text.y = element_text(size = 12),
      axis.title = element_text(size = 14, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
    ) +
    labs(
      x = "Model Type",
      y = "Mean Absolute Error (MAE)",
      title = title
    ) +
    scale_y_continuous(limits = limits,breaks = breaks)
  return(plot)
  
}

pairwise_cliff_delta <- function(data, model_list) {
  comparisons <- c()
  estimates <- c()
  
  # Loop through each pair and compute Cliff's delta
  for (pair in model_list) {
    result <- effsize::cliff.delta(data[[pair[1]]], data[[pair[2]]])
    comparisons <- c(comparisons, paste(pair, collapse = "_x_"))
    estimates <- c(estimates, abs(result$estimate))
  }
  
  # Combine into a data frame
  results_df <- data.frame(comparison = comparisons, estimate = estimates)
  
  return(results_df)
}