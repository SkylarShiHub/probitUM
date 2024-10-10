#' @title Produces the item characteristic curves
#' @description This function is used to generate the item characteristic curve for a given issue
#' @param rollnumber The roll number of the issue to be reviewed.
#' @param post_samples A list of posterior samples of parameters obtained from MCMC.
#' @importFrom Rcpp sourceCpp
#' @import ggplot2
#' @useDynLib probitUM
#' @return An item characteristic curve of the input issue
#' @examples
#' item_result = item_char_curve(rollnumber = 100, post_samples = post_samples)
#' @export
item_char_curve = function(rollnumber, post_samples){
  beta = as.matrix(post_samples$beta)
  alpha1 = as.matrix(post_samples$alpha1)
  alpha2 = as.matrix(post_samples$alpha2)
  delta1 = as.matrix(post_samples$delta1)
  delta2 = as.matrix(post_samples$delta2)

  col_index <- grep(paste0("^", rollnumber, "_"), colnames(alpha1))
  beta_samples = seq(min(beta), max(beta), length.out = 500)
  prob_mat = matrix(nrow = nrow(alpha1), ncol = length(beta_samples))

  for (i in (1:length(beta_samples))){
    term1 <- -alpha1[,col_index] * (beta_samples[i] - delta1[,col_index]) / sqrt(2)
    term2 <- -alpha2[,col_index] * (beta_samples[i] - delta2[,col_index]) / sqrt(2)
    prob_mat[,i] <- bvndvec(term1, term2, rep(0.5,length(alpha1[,col_index])))
  }
  means <- colMeans(prob_mat)
  ci_lower <- apply(prob_mat, 2, quantile, probs = 0.05)
  ci_upper <- apply(prob_mat, 2, quantile, probs = 0.95)

  plot_data <- data.frame(
    beta_samples = beta_samples,
    means = means,
    ci_lower = ci_lower,
    ci_upper = ci_upper
  )

  ggplot(plot_data, aes(x = beta_samples, y = means)) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper), fill = "grey80", alpha = 0.5) +
    geom_line(size = 1.5) +
    labs(x = expression(beta[i]), y = "Probability of voting 'Yes'",
         title = paste0("Issue No.", rollnumber)) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 22, face = "bold"),
      axis.title.x = element_text(size = 22),
      axis.title.y = element_text(size = 22),
      axis.text.x = element_text(size = 18),
      axis.text.y = element_text(size = 18)
    )

}
