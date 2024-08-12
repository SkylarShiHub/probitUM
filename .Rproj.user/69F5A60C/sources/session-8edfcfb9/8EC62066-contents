#' @title Calculate probabilities of a vote matrix
#' @description This function is used to get the probabilities of each vote result for the given members and issues
#' @param vote_m A vote matrix in which rows represent members and columns represent issues.
#' The entries can only be 0 (indicating ”No”), 1 (indicating ”Yes”), or NA (indicating missing data).
#' @param post_samples A list of posterior samples of parameters obtained from MCMC.
#' @importFrom Rcpp sourceCpp
#' @useDynLib probitUM
#' @return A matrix of probabilities corresponding to each vote
#' @examples
#' prob_mat = cal_prob(vote = house_votes_m, post_samples = post_samples)
#' @export

cal_prob <- function(vote, post_samples) {
  beta = as.matrix(post_samples$beta)
  alpha1 = as.matrix(post_samples$alpha1)
  alpha2 = as.matrix(post_samples$alpha2)
  delta1 = as.matrix(post_samples$delta1)
  delta2 = as.matrix(post_samples$delta2)
  n_samples <- nrow(beta)
  n_rows <- nrow(vote)
  n_cols <- ncol(vote)

  prob_mean <- matrix(0, nrow = n_rows, ncol = n_cols)

  for (k in 1:n_samples) {
    prob <- matrix(NA, nrow = n_rows, ncol = n_cols)

    for (i in 1:n_rows) {
      term1 <- -alpha1[k,] * (beta[k,i] - delta1[k,]) / sqrt(2)
      term2 <- -alpha2[k,] * (beta[k,i] - delta2[k,]) / sqrt(2)

      bvnd_vals <- bvndvec(term1, term2, rep(0.5, ncol(alpha1)))

      prob[i,] <- ifelse(is.na(vote[i,]), NA,
                         ifelse(vote[i,] == 1, bvnd_vals, 1 - bvnd_vals))
    }

    if (k == 1) {
      prob_mean <- prob
    } else {
      # cumulative average：new_mean = old_mean + (new_value - old_mean) / k
      prob_mean <- prob_mean + (prob - prob_mean)/k
    }
  }

  rownames(prob_mean) <- rownames(vote)
  colnames(prob_mean) <- colnames(vote)

  return(prob_mean)
}
