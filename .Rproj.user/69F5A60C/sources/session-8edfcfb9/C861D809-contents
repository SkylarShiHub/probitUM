#' @title Get median rank plots
#' @description Generates the median rank plot for all members based on posterior samples
#' @param beta A matrix of posterior samples of beta obtained from MCMC
#' @param col_blue A character string indicating whether the "D" party should be colored blue or "R".
#' @importFrom ggplot2 ggplot geom_point scale_color_manual scale_shape_manual labs theme_minimal theme element_text element_blank aes
#' @importFrom ggrepel geom_text_repel
#' @importFrom dplyr arrange desc top_n slice dense_rank
#' @importFrom magrittr %>%
#' @importFrom RColorBrewer brewer.pal
#' @importFrom graphics par
#' @return A list containing a ggplot object of the rank plot and a data frame of sorted rank data.
#' @examples
#' rank_results = median_rank_plot(beta = post_samples$beta, col_blue = "D")
#' @export
median_rank_plot = function(beta, col_blue = "D"){
  rank_matrix <- matrix(0, nrow = nrow(beta), ncol = ncol(beta))
  for (i in 1:nrow(beta)) {
    rank_matrix[i, ] <- rank(-beta[i, ])
  }
  # find the median rank of each member
  rank_median <- apply(rank_matrix, 2, median)

  all_members = gsub("_beta", "", colnames(post_samples$beta))
  party_vector <- ifelse(grepl("\\(D", all_members), "D",
                         ifelse(grepl("\\(R", all_members), "R",
                                ifelse(grepl("\\(I", all_members), "I", NA)))
  name_vector <- sub(" \\(.*\\)", "", all_members)
  rank_data = data.frame(names = name_vector, party = party_vector, median = rank_median)
  sorted_rank_data <- rank_data %>% arrange(desc(median))
  sorted_rank_data$party <- factor(sorted_rank_data$party)
  sorted_rank_data$rank <- dense_rank(-sorted_rank_data$median)

  # set up party colors
  colors <- brewer.pal(12, "Paired")
  if (col_blue == "D") {
    party_colors <- c("D" = colors[1], "R" = colors[5], "I" = "grey")
    party_shapes <- c("D" = 19, "R" = 17, "I" = 18)
  } else if (col_blue == "R") {
    party_colors <- c("D" = colors[5], "R" = colors[1], "I" = "grey")
    party_shapes <- c("D" = 17, "R" = 19, "I" = 18)
  }

  # labels for top & bottom five points
  top_labels <- sorted_rank_data %>% top_n(5, wt = median)
  bottom_labels <- sorted_rank_data %>% slice((n() - 4):n())
  median_index <- which(sorted_rank_data$rank == median(sorted_rank_data$rank))
  median_label <- sorted_rank_data[median_index, ]
  rank_plot = ggplot(sorted_rank_data, aes(x = median, y = rank, color = party, shape = party)) +
    geom_point(size = 2.2, alpha = 0.6) +
    geom_text_repel(data = top_labels, aes(label = names),
                    size = 3, col = "black") +
    geom_text_repel(data = bottom_labels, aes(label = names),
                    size = 3, col = "black") +
    geom_text_repel(data = median_label, aes(label = names),
                    size = 3, col = "black") +
    scale_color_manual(values = party_colors, labels = c("D", "R", "I")) +
    scale_shape_manual(values = party_shapes) +
    # scale_color_manual(values = party_colors) +
    labs(x = "Median", y = "Rank") +
    scale_y_reverse() +
    theme_minimal() +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())

  return(list(rank_plot = rank_plot, rank_data = sorted_rank_data))
}
