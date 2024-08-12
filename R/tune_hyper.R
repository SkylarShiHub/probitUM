#' @title generate posterior samples
#' @description This function is used to get samples of all the parameters from their posterior distributions
#' @param hyperparams A list of hyperparameter values. beta_mean is a value for the prior mean of beta.
#' beta_var is a value for the variance of beta. alpha_mean is a vector of 2 components representing the prior
#' means of alpha1 and alpha2. alpha_var is a value for the variance of alpha1 and alpha2.
#' delta_mean is a vector of 2 components representing the prior means of delta1 and delta2.
#' delta_var is a value for the variance of delta1 and delta2.
#' @param n_member A value representing the number of members to be simulated
#' @param n_issue A value representing the number of issuess to be simulated
#' @importFrom MASS mvrnorm
#' @importFrom truncnorm rtruncnorm
#' @importFrom Rcpp sourceCpp
#' @useDynLib probitUM
#' @return A list containing:
#' - `samples`: A list containing all sampled parameters from given distributions.
#' - `probability`: A matrix of probabilities of voting "Yes"
#' - `histogram`: A histogram of the density of probabilities.
#' @examples
#' hyperparams = list(beta_mean = 0, beta_var = 1, alpha_mean = c(0, 0),
#'                    alpha_var = 25, delta_mean = c(-2, 10), delta_var = 10)
#' tune_results = tune_hyper(hyperparams, n_member = 100, n_issue = 100)
#' @export
tune_hyper <- function(hyperparams = hyperparams, n_member, n_issue) {
  samples <- matrix(0, nrow = n_issue, ncol = 4)
  for (i in 1:n_issue) {
    if (runif(1) < 0.5) {
      alpha_j1 <- rtruncnorm(1, a = 0, b = Inf, mean = hyperparams$alpha_mean[1], sd = sqrt(hyperparams$alpha_var))
      alpha_j2 <- rtruncnorm(1, a = -Inf, b = 0, mean = hyperparams$alpha_mean[2], sd = sqrt(hyperparams$alpha_var))
      delta_j <- mvrnorm(1, hyperparams$delta_mean, matrix(c(hyperparams$delta_var, 0, 0, hyperparams$delta_var), nrow = 2))
    } else {
      alpha_j1 <- rtruncnorm(1, a = -Inf, b = 0, mean = hyperparams$alpha_mean[1], sd = sqrt(hyperparams$alpha_var))
      alpha_j2 <- rtruncnorm(1, a = 0, b = Inf, mean = hyperparams$alpha_mean[2], sd = sqrt(hyperparams$alpha_var))
      delta_j <- mvrnorm(1, -hyperparams$delta_mean, matrix(c(hyperparams$delta_var, 0, 0, hyperparams$delta_var), nrow = 2))
    }

    samples[i, ] <- c(alpha_j1, alpha_j2, delta_j)
  }
  beta = rnorm(n_member,hyperparams$beta_mean, sqrt(hyperparams$beta_var))
  samples = list(beta = beta, alpha1 = samples[,1], alpha2 = samples[,2],
                 delta1 = samples[,3], delta2 = samples[,4])
  mat <- matrix(1, nrow = n_member, ncol = n_issue)
  probability = get_prob_mat(mat, samples$beta, samples$alpha1, samples$alpha2,
                         samples$delta1, samples$delta2)
  mu_string <- paste("(", paste(hyperparams$delta_mean, collapse = ", "), ")", sep = "")
  histogram <- hist(probability, freq = FALSE,
                    xlab = expression(theta[list(i,j)]),
                    ylim = c(0,10),
                    main = bquote("Histogram of " * .(n_member*n_issue) *
                                    " draws with " ~
                                    mu == .(mu_string) * "," ~
                                    omega^2 == .(hyperparams$alpha_var) * "," ~
                                    kappa^2 == .(hyperparams$delta_var)),
                    border = "black", col = NA, lwd = 1,cex.lab = 1.2,
                    panel.first = grid())
  return(list(samples = samples, probability = probability, histogram = histogram))
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
