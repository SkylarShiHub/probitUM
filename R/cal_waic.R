#' @title Calculate WAIC
#' @description This function is used to get the WAIC value of the probit unfolding model
#' @param vote_m A vote matrix in which rows represent members and columns represent issues.
#' The entries can only be 0 (indicating ”No”), 1 (indicating ”Yes”), or NA (indicating missing data).
#' @param post_samples A list of posterior samples of parameters obtained from MCMC.
#' @importFrom Rcpp sourceCpp
#' @useDynLib probitUM
#' @return The WAIC value of this probit unfoldinng model
#' @examples
#' waic_val = cal_waic(vote = house_votes_m, post_samples = post_samples)
#' @export
cal_waic <- function(vote, post_samples) {

  sum_matrix <- matrix(nrow = nrow(vote), ncol = nrow(post_samples$beta))
  sum_matrix <- sapply(1:nrow(post_samples$beta), function(i) {
  cal_row_likelihood(vote, post_samples$beta[i, ], post_samples$alpha1[i, ],
                       post_samples$alpha2[i, ], post_samples$delta1[i, ], post_samples$delta2[i, ])
  })

  # log-sum-exp
  max_log_likelihood <- apply(sum_matrix, 1, max)
  sum_exp_diff <- rowSums(exp(sum_matrix - max_log_likelihood), na.rm = TRUE)
  log_sum_exp <- max_log_likelihood + log(sum_exp_diff) - log(nrow(post_samples$beta))

  term1 <- sum(log_sum_exp)
  term2 <- sum(apply(sum_matrix, 1, var))
  WAIC <- -2 * (term1 - term2)
  return(WAIC)
}

#' @title calculate the log-likelihood of each row (input vectors)
cal_row_likelihood <- function(vote, beta, alpha1, alpha2, delta1, delta2) {
  prob_matrix <- get_prob_mat(vote, as.numeric(beta), as.numeric(alpha1),
                          as.numeric(alpha2), as.numeric(delta1), as.numeric(delta2))
  prob_matrix[prob_matrix < 1e-09] <- 1e-09
  prob_matrix[prob_matrix > 1 - 1e-09] <- 1 - 1e-09
  row_sums <- rowSums(log(prob_matrix), na.rm = TRUE)
  return(row_sums)
}

#' @title calculate probability matrix of a given set of parameters
get_prob_mat <- function(vote, beta, alpha1, alpha2, delta1, delta2) {

  prob <- matrix(NA, nrow = nrow(vote), ncol = ncol(vote))

  for (j in 1:ncol(vote)) {

    term1 <- -alpha1[j] * (beta - delta1[j]) / sqrt(2)
    term2 <- -alpha2[j] * (beta - delta2[j]) / sqrt(2)
    bvnd_vals <- bvndvec(term1, term2, rep(0.5,length(beta)))
    prob[, j] <- ifelse(is.na(vote[, j]), NA,
                        ifelse(vote[, j] == 1, bvnd_vals, 1 - bvnd_vals))
  }
  rownames(prob) <- rownames(vote)
  colnames(prob) <- colnames(vote)
  return(prob)
}
