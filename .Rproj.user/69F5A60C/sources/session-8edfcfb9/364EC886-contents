#' @title generate posterior samples
#' @description This function is used to get samples of all the parameters from their posterior distributions
#' @param vote_m A vote matrix in which rows represent members and columns represent issues.
#' The entries can only be 0 (indicating ”No”), 1 (indicating ”Yes”), or NA (indicating missing data).
#' @param hyperparams A list of hyperparameter values. beta_mean is a value for the prior mean of beta.
#' beta_var is a value for the variance of beta. alpha_mean is a vector of 2 components representing the prior
#' means of alpha1 and alpha2. alpha_var is a value for the variance of alpha1 and alpha2.
#' delta_mean is a vector of 2 components representing the prior means of delta1 and delta2.
#' delta_var is a value for the variance of delta1 and delta2.
#' @param num_iter The total number of iterations
#' @param start_iter The iteration number to start retaining data after the burn-in period
#' @param keep_iter The interval at which iterations are kept for posterior samples.
#' @param leg_pos_init A vector of initial values for all the positions of legislators. The
#' length of this vector should be equal to the number of iterations times the number of members
#' @param alpha_pos_init A vector of initial values for all the alpha parameters. This vector
#' should have a length of 2 times the number of iterations times the number of issues,
#' as each issue has two alpha parameters.
#' @param delta_pos_init A vector of initial values for all the delta parameters. The length
#' should be 2 times the number of iterations times the number of issues
#' @param y_star_m_1_init Matrices of initial values for the three auxiliary variables, with
#' dimensions equal to the number of members by the number of issues. Same for the other two parameters.
#' @param pos_ind The index of the reference member whose position is kept positive.
#' @param neg_ind The index of the reference member whose position is kept negative.
#' @param start_val A vector of all initial values for each parameter.
#' @param sample_beta A Boolean value indicating whether to sample beta in the model.
#' @importFrom Rcpp sourceCpp
#' @useDynLib probitUM
#' @return A list mainly containing:
#' - `beta`: A matrix of parameter draws for beta.
#' - `alpha1`: A matrix of parameter draws of alpha1.
#' - `alpha2`: A matrix of parameter draws of alpha2.
#' - `delta1`: A matrix of parameter draws of delta1.
#' - `delta2`: A matrix of parameter draws of delta2.
#' @examples
#' hyperparams = list(beta_mean = 0, beta_var = 1, alpha_mean = c(0, 0),
#'                    alpha_var = 25, delta_mean = c(-2, 10), delta_var = 10)
#' post_samples = sample_three_utility_probit(house_votes_m, hyperparams,
#'           num_iter = 400, start_iter = 200,
#'           keep_iter = 1, leg_pos_init = NULL, alpha_pos_init = NULL,
#'           delta_pos_init = NULL, y_star_m_1_init = NULL, y_star_m_2_init = NULL,
#'           y_star_m_3_init = NULL, pos_ind = 82, neg_ind = 0, start_val = NULL,
#'           sample_beta = T)
#' @export
sample_three_utility_probit <- function(
    vote_m, hyperparams, num_iter = 20000, start_iter = 10000, keep_iter = 1,
    leg_pos_init = NULL, alpha_pos_init = NULL, delta_pos_init = NULL,
    y_star_m_1_init = NULL, y_star_m_2_init = NULL, y_star_m_3_init = NULL,
    pos_ind = 0, neg_ind = 0, start_val = NULL, sample_beta = T) {

  total_iter = (num_iter - start_iter) %/% keep_iter
  init_info <- init_data_rcpp(
    vote_m, leg_pos_init, alpha_pos_init, delta_pos_init,
    y_star_m_1_init, y_star_m_2_init, y_star_m_3_init, total_iter)

  if (!is.null(start_val)) {
    init_info[[1]][1,] <- start_val
  }
  # c++
  draw_info <- sample_three_utility_probit_rcpp(
    vote_m, init_info[[1]], init_info[[2]], init_info[[3]], init_info[[4]],
    init_info[[5]], init_info[[6]], init_info[[7]],
    init_info[[8]], init_info[[9]], hyperparams$beta_mean, sqrt(hyperparams$beta_var),
    hyperparams$alpha_mean, diag(2) * hyperparams$alpha_var,
    hyperparams$delta_mean, diag(2) * hyperparams$delta_var,
    num_iter, start_iter, keep_iter, pos_ind - 1, neg_ind - 1, sample_beta)

  all_param_draw = draw_info[[1]]
  leg_names <- sapply(rownames(vote_m), function(name) {paste(name, "beta", sep = "_")})
  if (is.null(colnames(vote_m))) {
    colnames(vote_m) <- sapply(1:ncol(vote_m), function(i) {
      paste("vote", i, sep = "_")
    })
  } # no operation performed
  alpha_vote_names_1 <- sapply(colnames(vote_m), function(name) {
    paste(name, "alpha", "1", sep = "_")
  })
  alpha_vote_names_2 <- sapply(colnames(vote_m), function(name) {
    paste(name, "alpha", "2", sep = "_")
  })
  delta_vote_names_1 <- sapply(colnames(vote_m), function(name) {
    paste(name, "delta", "1", sep = "_")
  })
  delta_vote_names_2 <- sapply(colnames(vote_m), function(name) {
    paste(name, "delta", "2", sep = "_")
  })
  colnames(all_param_draw) <-
    c(leg_names, alpha_vote_names_1, alpha_vote_names_2, delta_vote_names_1, delta_vote_names_2)

  beta_list <- as.data.frame(all_param_draw[, leg_names])
  alpha1_list <- as.data.frame(all_param_draw[, alpha_vote_names_1])
  alpha2_list <- as.data.frame(all_param_draw[, alpha_vote_names_2])
  delta1_list <- as.data.frame(all_param_draw[, delta_vote_names_1])
  delta2_list <- as.data.frame(all_param_draw[, delta_vote_names_2])

  return(c(list(
    beta = beta_list,
    alpha1 = alpha1_list,
    alpha2 = alpha2_list,
    delta1 = delta1_list,
    delta2 = delta2_list),
    other_info = draw_info[-1]
  ))
}

#' @title initialize three auxiliary parameters y
init_y_star_m <- function(vote_m) {
  y_star_m_1 <- vote_m
  y_star_m_2 <- vote_m
  y_star_m_3 <- vote_m

  y_star_m_1[which(vote_m == 1)] = 0
  y_star_m_2[which(vote_m == 1)] = 1
  y_star_m_3[which(vote_m == 1)] = 0

  no_votes <- which(vote_m == 0)
  sample_upper <- rbinom(length(no_votes), size = 1, prob = 0.5)
  y_star_m_1[no_votes[which(sample_upper == 1)]] = 0
  y_star_m_2[no_votes[which(sample_upper == 1)]] = 0
  y_star_m_3[no_votes[which(sample_upper == 1)]] = 1

  y_star_m_1[no_votes[which(sample_upper == 0)]] = 1
  y_star_m_2[no_votes[which(sample_upper == 0)]] = 0
  y_star_m_3[no_votes[which(sample_upper == 0)]] = 0

  return(list(y_star_m_1, y_star_m_2, y_star_m_3))
}

#' @title initialize member and position parameters and get starting points
init_data_rcpp <- function(vote_m, leg_pos_init, alpha_pos_init, delta_pos_init,
                           y_star_m_1_init, y_star_m_2_init, y_star_m_3_init, total_iter) {

  if (!is.null(leg_pos_init)) {
    leg_pos_m <-
      matrix(leg_pos_init, nrow = total_iter, ncol = nrow(vote_m), byrow = T)
  } else {
    leg_pos_m <- matrix(0, nrow = total_iter, ncol = nrow(vote_m)) # if it is not imputed
  }

  if (!is.null(alpha_pos_init)) {
    alpha_pos_m <-
      matrix(t(alpha_pos_init), nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T) # vector of 2
  } else {
    alpha_pos_m <-
      matrix(rep(c(-1, 1), ncol(vote_m)),
             nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  }

  if (!is.null(delta_pos_init)) {
    delta_pos_m <-
      matrix(t(delta_pos_init), nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T) # vector of 2
  } else {
    delta_pos_m <-
      matrix(0, nrow = total_iter, ncol = 2 * ncol(vote_m), byrow = T)
  }

  if (!is.null(y_star_m_1_init)) {
    y_star_m_1 <- y_star_m_1_init
    y_star_m_2 <- y_star_m_2_init
    y_star_m_3 <- y_star_m_3_init
  } else {
    y_star_info <- init_y_star_m(vote_m)
    y_star_m_1 <- y_star_info[[1]]
    y_star_m_2 <- y_star_info[[2]]
    y_star_m_3 <- y_star_info[[3]]
  }

  all_params_draw <- cbind(leg_pos_m, alpha_pos_m, delta_pos_m) # initial values of beta, alpha, delta
  beta_start_ind = 0; # used in C++
  alpha_start_ind = nrow(vote_m);
  alpha_2_start_ind = alpha_start_ind + ncol(vote_m);
  delta_start_ind = alpha_2_start_ind + ncol(vote_m);
  delta_2_start_ind = delta_start_ind + ncol(vote_m);

  return(list(all_params_draw, y_star_m_1, y_star_m_2,
              y_star_m_3, beta_start_ind,
              alpha_start_ind, alpha_2_start_ind,
              delta_start_ind, delta_2_start_ind))
}
