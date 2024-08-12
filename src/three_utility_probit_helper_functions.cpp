#include <RcppArmadillo.h>
#include <armadillo>
#include <cmath>
#include <Rcpp.h>
#include <algorithm>
#include <RcppDist.h>
#include <mvtnorm.h>
#include <R_ext/Rdynload.h>
//Code from RcppTN: https://github.com/olmjo/RcppTN/blob/master/src/rtn1.cpp
#include "rtn1.h"

using namespace Rcpp;
using namespace arma;
using namespace std;

//[[Rcpp::depends(RcppArmadillo, RcppDist, mvtnorm)]]

const double pi2 = pow(datum::pi,2);
const double TWOPI = 6.283185307179586;

// bool isVoteValid(double vote) {
//     return !NumericVector::is_na(vote); 
// }

bool isVoteValid(double vote) {
    return is_finite(vote); 
}
arma::mat create_ar_1_m(
    double term_length, double rho,
    double tau) {

  arma::mat ar_1_kernel(term_length, term_length);
  for (int i = 0; i < term_length; i++) {
    for (int j = i; j < term_length; j++) {
      ar_1_kernel(i, j) = tau / (1 - pow(rho, 2)) * pow(rho, j - i);
      ar_1_kernel(j, i) = ar_1_kernel(i, j);
    }
  }
  return(ar_1_kernel);
}

arma::mat create_ar_1_m_chol(double term_length, double rho, double tau) {
  arma::mat ar_1_kernel_chol(term_length, term_length);
  if (rho == 0 || term_length == 1) {
    return(ar_1_kernel_chol.eye(term_length, term_length) * sqrt(tau / (1 - pow(rho, 2))));
  }
  for (int i = 0; i < term_length; i++) {
    for (int j = i; j < term_length; j++) {
      ar_1_kernel_chol(i, j) = sqrt(tau) * pow(rho, j - i);
    }
  }
  for (int j = 0; j < term_length; j++) {
    ar_1_kernel_chol(0, j) = ar_1_kernel_chol(0, j) / sqrt(1 - pow(rho, 2));
  }
  return(ar_1_kernel_chol);
}

arma::mat create_ar_1_m_inverse(double term_length, double rho, double tau) {
  arma::mat inv_m(term_length, term_length);
  if (rho == 0 || term_length == 1) {
    return(inv_m.eye(term_length, term_length) * (1 - pow(rho, 2)) / tau);
  }
  inv_m(0,0) = 1;
  inv_m(term_length - 1, term_length - 1) = 1;
  inv_m(0,1) = -rho;
  inv_m(term_length - 1, term_length - 2) = -rho;
  if (term_length == 2) {
    return(inv_m / tau);
  }
  for (int i = 1; i < term_length - 1; i++) {
    inv_m(i, i - 1) = -rho;
    inv_m(i, i) = 1 + pow(rho, 2);
    inv_m(i, i + 1) = -rho;
  }
  return(inv_m / tau);
}

arma::vec simulate_draw_from_ar_1_m_chol(
    double term_length, double rho, double tau,
    double mean) {

  arma::mat chol_m = create_ar_1_m_chol(term_length, rho, tau);
  arma::vec draw(term_length, fill::randn);
  return(chol_m.t() * draw + mean);
}

double sample_three_utility_probit_beta(
    rowvec y_star_m_1, rowvec y_star_m_3,
    rowvec alpha_v_1, rowvec alpha_v_2,
    rowvec delta_v_1, rowvec delta_v_2,
    double beta_mean, double beta_s) {

  y_star_m_1 = y_star_m_1 - alpha_v_1 % delta_v_1;
  y_star_m_3 = y_star_m_3 - alpha_v_2 % delta_v_2;
  double post_var = 1.0 / pow(beta_s, 2) +
    dot(alpha_v_1, alpha_v_1) + dot(alpha_v_2, alpha_v_2);
  double post_mean = beta_mean / pow(beta_s, 2) -
    dot(alpha_v_1, y_star_m_1) - dot(alpha_v_2, y_star_m_3);

  return(randn() / sqrt(post_var) + post_mean / post_var); // a value of beta
}

arma::vec sample_three_utility_probit_beta_gp(
    rowvec y_star_m_1, rowvec y_star_m_3,
    rowvec alpha_v_1, rowvec alpha_v_2,
    rowvec delta_v_1, rowvec delta_v_2,
    arma::uvec case_year, double rho) {

  int years_served = max(case_year) - min(case_year) + 1;
  arma::mat ar_1_m_inv = create_ar_1_m_inverse(years_served, rho, 1 - rho * rho);
  y_star_m_1 = y_star_m_1 - alpha_v_1 % delta_v_1;
  y_star_m_3 = y_star_m_3 - alpha_v_2 % delta_v_2;

  arma::vec post_mean(years_served, fill::zeros);
  for (int i = 0; i < case_year.n_elem; i++) {
    ar_1_m_inv(case_year(i), case_year(i)) +=
      alpha_v_1(i) * alpha_v_1(i) + alpha_v_2(i) * alpha_v_2(i);
    post_mean(case_year(i)) -=
      alpha_v_1(i) * y_star_m_1(i) + alpha_v_2(i) * y_star_m_3(i);
  }
  post_mean = solve(ar_1_m_inv, post_mean);
  return(rmvnorm(1, post_mean, ar_1_m_inv.i()).t());
}

double logit(double p) {
  return(log(p) - log(1 - p));
}

double inv_logit(double z) {
  return(1.0 / (1.0 + exp(-z)));
}

double sample_rho_pos_logit_gibbs(
    double rho, arma::vec ideal_pos_1_m,
    arma::uvec judge_start_ind, arma::uvec judge_end_ind,
    double rho_mean,
    double rho_sigma, double rho_sd) {

  double next_rho = inv_logit(logit(rho) + rho_sd * randn());
  double next_log_ll =
    d_truncnorm(next_rho, rho_mean, rho_sigma, 0, 1, 1) +
                  log(next_rho) + log(1 - next_rho);
  double prev_log_ll =
    d_truncnorm(rho, rho_mean, rho_sigma, 0, 1, 1) +
                  log(rho) + log(1 - rho);
  for (int i = 0; i < judge_start_ind.n_elem; i++) {
    rowvec pos_v = ideal_pos_1_m(span(judge_start_ind(i),
                                      judge_end_ind(i))).t();

    arma::mat prev_ar_1_m = create_ar_1_m(pos_v.n_elem, rho, 1 - rho * rho);
    prev_log_ll += as_scalar(dmvnorm(pos_v, zeros(pos_v.n_elem), prev_ar_1_m, true));

    arma::mat next_ar_1_m = create_ar_1_m(pos_v.n_elem, next_rho, 1 - next_rho * next_rho);
    next_log_ll += as_scalar(dmvnorm(pos_v, zeros(pos_v.n_elem), next_ar_1_m, true));
  }
  if (log(randu()) < next_log_ll - prev_log_ll) {
    return(next_rho);
  }
  return(rho);
}

arma::vec sample_three_utility_probit_matched_alpha(
    arma::vec y_star_m_1, arma::vec y_star_m_3,
    arma::vec beta_v, arma::vec delta_v,
    arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s) {

  arma::vec beta_diff_v_1 = beta_v - delta_v(0);
  arma::vec beta_diff_v_2 = beta_v - delta_v(1);

  arma::mat post_cov = alpha_cov_s.i();
  post_cov(0, 0) += dot(beta_diff_v_1, beta_diff_v_1);
  post_cov(1, 1) += dot(beta_diff_v_2, beta_diff_v_2);

  arma::vec post_mean = solve(alpha_cov_s, alpha_mean_v);
  post_mean(0) -= dot(beta_diff_v_1, y_star_m_1);
  post_mean(1) -= dot(beta_diff_v_2, y_star_m_3);
  post_mean = solve(post_cov, post_mean);

  double sample_order_up_prob =
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), false, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), true, true) +
    as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
  double sample_order_down_prob =
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), true, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), false, true) +
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));

  double log_sample_prob = sample_order_up_prob -
    (max(sample_order_up_prob, sample_order_down_prob) +
    log(1 + exp(min(sample_order_up_prob, sample_order_down_prob) -
                  max(sample_order_up_prob, sample_order_down_prob))));
  double match_var = (log(randu()) < log_sample_prob) * 2 - 1;

  arma::vec out_v(3);
  if (match_var == 1) {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)), // one random number from truncated normal 
                    0, datum::inf);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)),
                    -datum::inf, 0);
  } else {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)),
                    -datum::inf, 0);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)),
                    0, datum::inf);
  }
  out_v(2) = match_var;

  return(out_v); // out_v[3] = {alpha_j_1, alpha_j_2, match_var}
}

arma::vec sample_three_utility_probit_matched_alpha_flip(
    arma::vec y_star_m_1, arma::vec y_star_m_3,
    arma::vec beta_v, arma::vec delta_v,
    arma::vec alpha_mean_v, arma::mat alpha_cov_s,
    arma::vec delta_mean_v, arma::mat delta_cov_s) {

  arma::vec beta_diff_v_1 = beta_v - delta_v(0);
  arma::vec beta_diff_v_2 = beta_v - delta_v(1);

  arma::mat post_cov = alpha_cov_s.i();
  post_cov(0, 0) += dot(beta_diff_v_1, beta_diff_v_1);
  post_cov(1, 1) += dot(beta_diff_v_2, beta_diff_v_2);

  arma::vec post_mean = solve(alpha_cov_s, alpha_mean_v);
  post_mean(0) -= dot(beta_diff_v_1, y_star_m_1);
  post_mean(1) -= dot(beta_diff_v_2, y_star_m_3);
  post_mean = solve(post_cov, post_mean);

  double sample_order_up_prob =
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), false, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), true, true) +
    as_scalar(dmvnorm(delta_v.t(), delta_mean_v, delta_cov_s, true));
  double sample_order_down_prob =
    R::pnorm(0, post_mean(0), sqrt(1.0 / post_cov(0,0)), true, true) +
    R::pnorm(0, post_mean(1), sqrt(1.0 / post_cov(1,1)), false, true) +
    as_scalar(dmvnorm(delta_v.t(), -delta_mean_v, delta_cov_s, true));

  double log_sample_prob = sample_order_up_prob -
    (max(sample_order_up_prob, sample_order_down_prob) +
    log(1 + exp(min(sample_order_up_prob, sample_order_down_prob) -
                  max(sample_order_up_prob, sample_order_down_prob))));
  double match_var = (log(randu()) < log_sample_prob) * 2 - 1;

  arma::vec out_v(3);
  if (match_var == -1) {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)), // one random number from truncated normal 
                    0, datum::inf);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)),
                    -datum::inf, 0);
  } else {
    out_v(0) = rtn1(post_mean(0), 1.0 / sqrt(post_cov(0, 0)),
                    -datum::inf, 0);
    out_v(1) = rtn1(post_mean(1), 1.0 / sqrt(post_cov(1, 1)),
                    0, datum::inf);
  }
  out_v(2) = match_var;

  return(out_v);
}
arma::vec sample_three_utility_probit_matched_delta(
    arma::vec y_star_m_1, arma::vec y_star_m_3,
    arma::vec alpha_v, arma::vec beta_v, double match_var,
    arma::vec delta_mean_v, arma::mat delta_cov_s) {

  y_star_m_1 += alpha_v(0) * beta_v;
  y_star_m_3 += alpha_v(1) * beta_v;

  arma::mat post_cov = beta_v.n_elem *
    diagmat(alpha_v) * diagmat(alpha_v) +
    delta_cov_s.i();
  arma::vec post_mean = match_var * solve(delta_cov_s, delta_mean_v);
  post_mean(0) += accu(alpha_v(0) * y_star_m_1);
  post_mean(1) += accu(alpha_v(1) * y_star_m_3);
  return(rmvnorm(1, solve(post_cov, post_mean),
                 post_cov.i()).t()); // {delta_j_1,delta_j_2}
}

arma::vec sample_three_utility_probit_matched_delta_flip(
    arma::vec y_star_m_1, arma::vec y_star_m_3,
    arma::vec alpha_v, arma::vec beta_v, double match_var,
    arma::vec delta_mean_v, arma::mat delta_cov_s) {

  y_star_m_1 += alpha_v(0) * beta_v;
  y_star_m_3 += alpha_v(1) * beta_v;

  arma::mat post_cov = beta_v.n_elem *
    diagmat(alpha_v) * diagmat(alpha_v) +
    delta_cov_s.i();
  arma::vec post_mean = match_var * solve(delta_cov_s, delta_mean_v);
  post_mean(0) += accu(alpha_v(0) * y_star_m_1);
  post_mean(1) += accu(alpha_v(1) * y_star_m_3);
  return(rmvnorm(1, -solve(post_cov, post_mean),
                 post_cov.i()).t()); // {delta_j_1,delta_j_2}
}

arma::vec sample_y_star_m_na(double mean_m_1, double mean_m_2) {
  arma::vec out_v(3, fill::randn);
  out_v(0) -= mean_m_1;
  out_v(2) -= mean_m_2;
  return(out_v);
}

arma::vec sample_y_star_m_yea(arma::vec y_star_yea, double mean_m_1, double mean_m_2) {

  y_star_yea(0) =
    rtn1(-mean_m_1, 1, -datum::inf, y_star_yea(1));
  y_star_yea(1) =
    rtn1(0, 1, max(y_star_yea(0), y_star_yea(2)), datum::inf);
  y_star_yea(2) =
    rtn1(-mean_m_2, 1, -datum::inf, y_star_yea(1));
  return(y_star_yea);
}

arma::vec sample_y_star_m_no(arma::vec y_star_no, double mean_m_1, double mean_m_2) {

  if (y_star_no(2) < y_star_no(1)) {
    y_star_no(0) =
      rtn1(-mean_m_1, 1, y_star_no(1), datum::inf);
  } else {
    y_star_no(0) = randn() - mean_m_1;
  }

  y_star_no(1) =
    rtn1(0, 1, -datum::inf, max(y_star_no(0), y_star_no(2)));

  if (y_star_no(0) < y_star_no(1)) {
    y_star_no(2) =
      rtn1(-mean_m_2, 1, y_star_no(1), datum::inf);
  } else {
    y_star_no(2) = randn() - mean_m_2;
  }
  return(y_star_no);
}

arma::vec sample_y_star_m(arma::vec y_star_vec, double vote, double alpha_1, double alpha_2,
                    double leg_pos, double delta_1, double delta_2) {

  arma::vec out_vec(3);
  double mean_m_1 = alpha_1 * (leg_pos - delta_1);
  double mean_m_2 = alpha_2 * (leg_pos - delta_2);
  if (vote == 1) {
    out_vec = sample_y_star_m_yea(y_star_vec, mean_m_1, mean_m_2);
  } else {
    out_vec = sample_y_star_m_no(y_star_vec, mean_m_1, mean_m_2);
  }
  return(out_vec);
}

// BVND calculates the probability that X > DH and Y > DK.
// Note: Prob( X < DH, Y < DK ) = BVND( -DH, -DK, R )
// Code and description is adopted from tvpack.f in the
// mvtnorm package with help from ChatGPT
// [[Rcpp::export]]
double bvnd(double DH, double DK, double R) {

  arma::vec x;
  arma::vec w;
  // double as = 0.0;
  // double a = 0.0;
  double b = 0.0;
  // double c = 0.0;
  // double d = 0.0;
  double rs = 0.0;
  double xs = 0.0;
  double bvn = 0.0;
  // double sn = 0.0;
  // double asr = 0.0;
  double h = DH;
  double k = DK;
  double hk = h * k;

  if (std::abs(R) < 0.3) {
    x = {-0.9324695142031522,-0.6612093864662647,
         -0.2386191860831970};
    w = {0.1713244923791705,
         0.3607615730481384, 0.4679139345726904};
  } else if (std::abs(R) < 0.75) {
    x = {-0.9815606342467191, -0.9041172563704750,
         -0.7699026741943050, -0.5873179542866171,
         -0.3678314989981802, -0.1252334085114692};
    w = {0.4717533638651177e-01, 0.1069393259953183,
         0.1600783285433464, 0.2031674267230659,
         0.2334925365383547, 0.2491470458134029};
  } else {
    x = {-0.9931285991850949, -0.9639719272779138,
         -0.9122344282513259, -0.8391169718222188,
         -0.7463319064601508, -0.6360536807265150,
         -0.5108670019508271, -0.3737060887154196,
         -0.2277858511416451, -0.7652652113349733e-01};
    w = {0.1761400713915212e-01, 0.4060142980038694e-01,
         0.6267204833410906e-01, 0.8327674157670475e-01,
         0.1019301198172404, 0.1181945319615184,
         0.1316886384491766, 0.1420961093183821,
         0.1491729864726037, 0.1527533871307259};
  }

  if (std::abs(R) < 0.925) {
    if (std::abs(R) > 0.0) {
      double hs = (h * h + k * k) / 2.0;
      double asr = std::asin(R);
      for (int i = 0; i < x.n_elem; ++i) {
        for (int is = -1; is <= 1; is += 2) {
          double sn = std::sin(asr * (is * x[i] + 1) / 2.0);
          bvn += w[i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
        }
      }
      bvn = bvn * asr / (2.0 * TWOPI);
    }
    bvn += R::pnorm(-h, 0, 1,-datum::inf, false) * R::pnorm(-k, 0, 1,-datum::inf, false);
  } else {
    if (R < 0.0) {
      k = -k;
      hk = -hk;
    }
    if (std::abs(R) < 1.0) {
      double as = (1.0 - R) * (1.0 + R);
      double a = std::sqrt(as);
      double bs = std::pow(h - k, 2);
      double c = (4.0 - hk) / 8.0;
      double d = (12.0 - hk) / 16.0;
      double asr = -(bs / as + hk) / 2.0;
      if (asr > -100.0) {
        bvn = a * std::exp(asr) *
          (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 +
          c * d * as * as / 5);
      }
      if (-hk < 100) {
        b = sqrt(bs);
        bvn = bvn - exp(-hk/2) * sqrt(TWOPI) * R::pnorm(-b/a, 0, 1,-datum::inf, false) * b
                * (1 - c * bs * (1 - d * bs/5) / 3);
      }
      a = a / 2;
      for (int i = 0; i < x.n_elem; i++) {
        for (int is = -1; is <= 1; is += 2) {
          xs = pow(a * (is * x[i] + 1), 2);
          rs = sqrt(1 - xs);
          asr = -(bs/xs + hk) / 2;
          if (asr > -100) {
            bvn = bvn + a * w[i] * exp(asr)
                   * (exp(-hk * (1 - rs) / (2 * (1 + rs)))/rs
                        - (1 + c * xs * (1 + d * xs)));
            }
          }
        }
        bvn = -bvn/TWOPI;
    }
    if (R > 0) {
      bvn = bvn + R::pnorm(-std::max(h, k), 0, 1,-datum::inf, false);
    } else {
      bvn = -bvn;
      if (k > h) {
        bvn = bvn + R::pnorm(k, 0, 1,-datum::inf, false) - R::pnorm(h, 0, 1,-datum::inf, false);
      }
    }
  }
  return(bvn);
}

// A vector version bvnd, whether beta is a vec or alpha,delta are vectors

// [[Rcpp::export]]
NumericVector bvndvec(NumericVector DH, NumericVector DK, NumericVector R) {
  int n = DH.size();
  NumericVector bvn(n);

  for (int idx = 0; idx < n; ++idx) {
    double h = DH[idx];
    double k = DK[idx];
    double r = R[idx];
    double hk = h * k;
    double bvn_single = 0.0;
    double rs = 0.0;
    double xs = 0.0;

    std::vector<double> x;
    std::vector<double> w;

    if (std::abs(r) < 0.3) {
      x = {-0.9324695142031522, -0.6612093864662647, -0.2386191860831970};
      w = {0.1713244923791705, 0.3607615730481384, 0.4679139345726904};
    } else if (std::abs(r) < 0.75) {
      x = {-0.9815606342467191, -0.9041172563704750, -0.7699026741943050,
           -0.5873179542866171, -0.3678314989981802, -0.1252334085114692};
      w = {0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
           0.2031674267230659, 0.2334925365383547, 0.2491470458134029};
    } else {
      x = {-0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
           -0.8391169718222188, -0.7463319064601508, -0.6360536807265150,
           -0.5108670019508271, -0.3737060887154196, -0.2277858511416451,
           -0.07652652113349733};
      w = {0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
           0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
           0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
           0.1527533871307259};
    }

    if (std::abs(r) < 0.925) {
      if (std::abs(r) > 0.0) {
        double hs = (h * h + k * k) / 2.0;
        double asr = std::asin(r);
        for (size_t i = 0; i < x.size(); ++i) {
          for (int is = -1; is <= 1; is += 2) {
            double sn = std::sin(asr * (is * x[i] + 1) / 2.0);
            bvn_single += w[i] * std::exp((sn * hk - hs) / (1.0 - sn * sn));
          }
        }
        bvn_single = bvn_single * asr / (2.0 * M_PI);
      }
      bvn_single += R::pnorm(-h, 0, 1, true, false) * R::pnorm(-k, 0, 1, true, false);
    } else {
      if (r < 0.0) {
        k = -k;
        hk = -hk;
      }
      if (std::abs(r) < 1.0) {
        double as = (1.0 - r) * (1.0 + r);
        double a = std::sqrt(as);
        double bs = std::pow(h - k, 2);
        double c = (4.0 - hk) / 8.0;
        double d = (12.0 - hk) / 16.0;
        double asr = -(bs / as + hk) / 2.0;
        if (asr > -100.0) {
          bvn_single = a * std::exp(asr) *
            (1.0 - c * (bs - as) * (1.0 - d * bs / 5.0) / 3.0 +
            c * d * as * as / 5);
        }
        if (-hk < 100) {
          double b = sqrt(bs);
          bvn_single -= std::exp(-hk / 2) * std::sqrt(2 * M_PI) * R::pnorm(-b / a, 0, 1, true, false) * b
                  * (1 - c * bs * (1 - d * bs / 5) / 3);
        }
        a = a / 2;
        for (size_t i = 0; i < x.size(); i++) {
          for (int is = -1; is <= 1; is += 2) {
            xs = std::pow(a * (is * x[i] + 1), 2);
            rs = std::sqrt(1 - xs);
            asr = -(bs / xs + hk) / 2;
            if (asr > -100) {
              bvn_single += a * w[i] * std::exp(asr)
                     * (std::exp(-hk * (1 - rs) / (2 * (1 + rs))) / rs
                          - (1 + c * xs * (1 + d * xs)));
              }
            }
          }
          bvn_single = -bvn_single / (2 * M_PI);
      }
      if (r > 0) {
        bvn_single = bvn_single + R::pnorm(-std::max(h, k), 0, 1, true, false);
      } else {
        bvn_single = -bvn_single;
        if (k > h) {
          bvn_single = bvn_single + R::pnorm(k, 0, 1, true, false) - R::pnorm(h, 0, 1, true, false);
        }
      }
    }
    bvn[idx] = bvn_single;
  }

  return bvn;
}

// final one
// [[Rcpp::export]]
List sample_three_utility_probit_rcpp(
  arma::mat vote_m, arma::mat all_param_draws, arma::mat y_star_m_1, arma::mat y_star_m_2, arma::mat y_star_m_3,
  int leg_start_ind, int alpha_v_1_start_ind, int alpha_v_2_start_ind,
  int delta_v_1_start_ind, int delta_v_2_start_ind,
  double leg_mean, double leg_sd, arma::vec alpha_mean_v, arma::mat alpha_cov_s,
  arma::vec delta_mean_v, arma::mat delta_cov_s, int num_iter, int start_iter,
  int keep_iter, int pos_ind, int neg_ind, bool sample_beta) {

  // arma::mat L_0(num_iter,vote_m.n_cols);
  // arma::mat L_1(num_iter,vote_m.n_cols);
  arma::vec current_param_val_v = all_param_draws.row(0).t(); // take the first row (initial values) and transpose
  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }
    // sample y_star
    for (int j = 0; j < vote_m.n_rows; j++) {
      for (int k = 0; k < vote_m.n_cols; k++) {
        if (!is_finite(vote_m(j, k))) {
          continue;
        }
        arma::vec y_star_vec = {y_star_m_1(j, k),
                          y_star_m_2(j, k),
                          y_star_m_3(j, k)}; // Assign the latest value to _star_vec and proceed to the next sampling
        arma::vec out_v = sample_y_star_m(
          y_star_vec, vote_m(j, k),
          current_param_val_v(alpha_v_1_start_ind + k), // alpha_v_1_start_ind is a variable representing the starting index of alpha_1 in this vector
          current_param_val_v(alpha_v_2_start_ind + k),
          current_param_val_v(leg_start_ind + j),
          current_param_val_v(delta_v_1_start_ind + k),
          current_param_val_v(delta_v_2_start_ind + k));
        y_star_m_1(j, k) = out_v(0);
        y_star_m_2(j, k) = out_v(1);
        y_star_m_3(j, k) = out_v(2);  // The values of y_star are not stored for each iteration i
      }
    }
    // sample beta
    if (sample_beta) {
      for (unsigned int j = 0; j < vote_m.n_rows; j++) {
        arma::uvec current_ind = {j};
        arma::uvec interested_inds = find_finite(vote_m.row(j).t());
        current_param_val_v(leg_start_ind + j) =
          sample_three_utility_probit_beta(
            y_star_m_1.submat(current_ind, interested_inds),
            y_star_m_3.submat(current_ind, interested_inds),
            current_param_val_v(alpha_v_1_start_ind + interested_inds).t(),
            current_param_val_v(alpha_v_2_start_ind + interested_inds).t(),
            current_param_val_v(delta_v_1_start_ind + interested_inds).t(),
            current_param_val_v(delta_v_2_start_ind + interested_inds).t(),
            leg_mean, leg_sd);
      }
    }
    // sample alpha
    arma::vec match_var_v(vote_m.n_cols);
    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      arma::uvec current_ind = {j};
      arma::uvec interested_inds = find_finite(vote_m.col(j));
      arma::vec delta_v = {current_param_val_v(delta_v_1_start_ind + j),
                     current_param_val_v(delta_v_2_start_ind + j)};
      arma::vec out_v =
        sample_three_utility_probit_matched_alpha(
          y_star_m_1.submat(interested_inds, current_ind),
          y_star_m_3.submat(interested_inds, current_ind),
          current_param_val_v(leg_start_ind + interested_inds),
          delta_v, alpha_mean_v, alpha_cov_s,
          delta_mean_v, delta_cov_s);

      current_param_val_v(alpha_v_1_start_ind + j) = out_v(0);
      current_param_val_v(alpha_v_2_start_ind + j) = out_v(1);
      match_var_v(j) = out_v(2);
    }
    // sample delta
    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      arma::uvec current_ind = {j};
      arma::uvec interested_inds = find_finite(vote_m.col(j));
      arma::vec alpha_v = {current_param_val_v(alpha_v_1_start_ind + j),
                     current_param_val_v(alpha_v_2_start_ind + j)};
      arma::vec out_v =
        sample_three_utility_probit_matched_delta(
          y_star_m_1.submat(interested_inds, current_ind),
          y_star_m_3.submat(interested_inds, current_ind),
          alpha_v, current_param_val_v(leg_start_ind + interested_inds),
          match_var_v(j), delta_mean_v, delta_cov_s);
      current_param_val_v(delta_v_1_start_ind + j) = out_v(0);
      current_param_val_v(delta_v_2_start_ind + j) = out_v(1);
    }
    // flipping signs
    if ((i+1) % 5 == 0){  // each 5 iterations
      if (randu() < 1){ // switch between method 1 and 2
        
        // 1. Metropolis-Hastings for just flipping the signs
        for (int j = 0; j < vote_m.n_cols; j++){
          double L_orig = 0.0;
          double L_new = 0.0;

          // if (j == 1 && i < 50) {
          // Rcout << "iter = " << i+1 << endl;
          // Rcout << "j = " << j << endl;
          // Rcout << "current_alpha_1: " << current_param_val_v(alpha_v_1_start_ind + j) << endl;
          // Rcout << "alpha_v_1_new: " << -current_param_val_v(alpha_v_1_start_ind + j) << endl;
          // Rcout << "current_alpha_2: " << current_param_val_v(alpha_v_2_start_ind + j) << endl;
          // Rcout << "alpha_v_2_new: " << -current_param_val_v(alpha_v_2_start_ind + j) << endl;
          // Rcout << "current_delta_1: " << current_param_val_v(delta_v_1_start_ind + j) << endl;
          // Rcout << "delta_v_1_new: " << -current_param_val_v(delta_v_1_start_ind + j) << endl;
          // Rcout << "current_delta_2: " << current_param_val_v(delta_v_2_start_ind + j) << endl;
          // Rcout << "delta_v_2_new: " << -current_param_val_v(delta_v_2_start_ind + j) << endl;
          // }

          for (int k = 0; k < vote_m.n_rows; k++) {
            if (!isVoteValid(vote_m(k, j))) continue;  // Skip NA votes
            // if (j == 1 && i == 49) {
            //   Rcout << "vote_m(" << k << ", " << j << "): " << vote_m(k, j) << endl;
            // }
            arma::vec param_now = {current_param_val_v(leg_start_ind + k),
                         current_param_val_v(alpha_v_1_start_ind + j),
                         current_param_val_v(alpha_v_2_start_ind + j),
                         current_param_val_v(delta_v_1_start_ind + j),
                         current_param_val_v(delta_v_2_start_ind + j)};
            arma::vec param_new = {current_param_val_v(leg_start_ind + k),
                         -current_param_val_v(alpha_v_1_start_ind + j),
                         -current_param_val_v(alpha_v_2_start_ind + j),
                         -current_param_val_v(delta_v_1_start_ind + j),
                         -current_param_val_v(delta_v_2_start_ind + j)};
            double p_orig = bvnd(-param_now(1) * (param_now(0) - param_now(3))/sqrt(2),
                       -param_now(2) * (param_now(0) - param_now(4))/sqrt(2), 0.5);
            double p_new = bvnd(-param_new(1) * (param_new(0) - param_new(3))/sqrt(2),
                     -param_new(2) * (param_new(0) - param_new(4))/sqrt(2), 0.5);         
            L_orig += log(p_orig) * vote_m(k, j) + log(1 - p_orig) * (1 - vote_m(k, j));
            L_new += log(p_new) * vote_m(k, j) + log(1 - p_new) * (1 - vote_m(k, j));
          }
          // L_0(i,j) = L_orig;
          // L_1(i,j) = L_new;
          double flip_prob = min(1.0, exp(L_new - L_orig));
          // if (j == 1 && i < 50){
          //   Rcout << "flip_prob: " << flip_prob << endl;
          // }
          if (randu() < flip_prob){
              current_param_val_v(alpha_v_1_start_ind + j) = -current_param_val_v(alpha_v_1_start_ind + j);
              current_param_val_v(alpha_v_2_start_ind + j) = -current_param_val_v(alpha_v_2_start_ind + j);
              current_param_val_v(delta_v_1_start_ind + j) = -current_param_val_v(delta_v_1_start_ind + j);
              current_param_val_v(delta_v_2_start_ind + j) = -current_param_val_v(delta_v_2_start_ind + j);
          }
        }

      } else{
        // 2. Metropolis-Hastings for resampling alpha and delta from the priors
        for (int j = 0; j < vote_m.n_cols; j++){
          double L_orig = 0.0;
          double L_new = 0.0;
          double alpha_v_1_new = 0.0;
          double alpha_v_2_new = 0.0;
          double delta_v_1_new = 0.0;
          double delta_v_2_new = 0.0;
          // arma::vec delta_v_new(2);
          if (current_param_val_v(alpha_v_1_start_ind + j) > 0){
            alpha_v_1_new = rtn1(0, sqrt(alpha_cov_s(0, 0)), -datum::inf, 0); 
            alpha_v_2_new = rtn1(0, sqrt(alpha_cov_s(0, 0)), 0, datum::inf);
            delta_v_1_new = rnorm(1, -delta_mean_v[0], sqrt(delta_cov_s(0,0)))[0]; 
            delta_v_2_new = rnorm(1, -delta_mean_v[1], sqrt(delta_cov_s(0,0)))[0];
            // delta_v_new = rmvnorm(1, -mu_0, sigma_delta).row(0);
          } else{
            alpha_v_1_new = rtn1(0, sqrt(alpha_cov_s(0, 0)), 0, datum::inf);
            alpha_v_2_new = rtn1(0, sqrt(alpha_cov_s(0, 0)), -datum::inf, 0);
            delta_v_1_new = rnorm(1, delta_mean_v[0], sqrt(delta_cov_s(0,0)))[0];
            delta_v_2_new = rnorm(1, delta_mean_v[1], sqrt(delta_cov_s(0,0)))[0];
            // delta_v_new = rmvnorm(1, mu_0, sigma_delta).row(0);
          }
          // if (j == 1 && i < 50) {
          //   Rcout << "iter = " << i+1 << endl;
          //   Rcout << "j = " << j << endl;
          //   Rcout << "current_alpha_1: " << current_param_val_v(alpha_v_1_start_ind + j) << endl;
          //   Rcout << "alpha_v_1_new: " << alpha_v_1_new << endl;
          //   Rcout << "current_alpha_2: " << current_param_val_v(alpha_v_2_start_ind + j) << endl;
          //   Rcout << "alpha_v_2_new: " << alpha_v_2_new << endl;
          //   Rcout << "current_delta_1: " << current_param_val_v(delta_v_1_start_ind + j) << endl;
          //   Rcout << "delta_v_1_new: " << delta_v_1_new << endl;
          //   Rcout << "current_delta_2: " << current_param_val_v(delta_v_2_start_ind + j) << endl;
          //   Rcout << "delta_v_2_new: " << delta_v_2_new << endl;

          // }

          

          for (int k = 0; k < vote_m.n_rows; k++) {
            if (!isVoteValid(vote_m(k, j))) continue; // Skip NA votes
            // if (j == 1 && i == 49) {
            //   Rcout << "vote_m(" << k << ", " << j << "): " << vote_m(k, j) << endl;
            // }

            // Rcout << "vote_m(" << k << ", " << j << "): " << vote_m(k, j) << endl;
            arma::vec param_now = {current_param_val_v(leg_start_ind + k),
                         current_param_val_v(alpha_v_1_start_ind + j),
                         current_param_val_v(alpha_v_2_start_ind + j),
                         current_param_val_v(delta_v_1_start_ind + j),
                         current_param_val_v(delta_v_2_start_ind + j)};
            arma::vec param_new = {current_param_val_v(leg_start_ind + k),
                         alpha_v_1_new, alpha_v_2_new,
                         delta_v_1_new, delta_v_2_new};

            double p_orig = bvnd(-param_now(1) * (param_now(0) - param_now(3))/sqrt(2),
                       -param_now(2) * (param_now(0) - param_now(4))/sqrt(2), 0.5);
            double p_new = bvnd(-param_new(1) * (param_new(0) - param_new(3))/sqrt(2),
                     -param_new(2) * (param_new(0) - param_new(4))/sqrt(2), 0.5);          
            L_orig += log(p_orig) * vote_m(k, j) + log(1 - p_orig) * (1 - vote_m(k, j));
            L_new += log(p_new) * vote_m(k, j) + log(1 - p_new) * (1 - vote_m(k, j));
          }
          
          // L_0(i,j) = L_orig;
          // L_1(i,j) = L_new;
          double flip_prob = min(1.0, exp(L_new - L_orig));
          // if (j == 1 && i < 50){
          //   Rcout << "flip_prob: " << flip_prob << endl;
          // }
          
          if (randu() < flip_prob){
              current_param_val_v(alpha_v_1_start_ind + j) = alpha_v_1_new;
              current_param_val_v(alpha_v_2_start_ind + j) = alpha_v_2_new;
              current_param_val_v(delta_v_1_start_ind + j) = delta_v_1_new;
              current_param_val_v(delta_v_2_start_ind + j) = delta_v_2_new;
          }
        }
      }
    }
     
    if (pos_ind > -1 && (current_param_val_v(leg_start_ind + pos_ind) < 0)) {
      current_param_val_v = -current_param_val_v;
    }

    if (neg_ind > -1 && pos_ind < 0 && (current_param_val_v(leg_start_ind + neg_ind) > 0)) {
      current_param_val_v = -current_param_val_v;
    }

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m_1") = y_star_m_1,
                      Named("y_star_m_2") = y_star_m_2,
                      Named("y_star_m_3") = y_star_m_3);
                      // Named("L_0") = L_0,
                      // Named("L_1") = L_1);
}

arma::vec adjust_all_judge_ideology(
    arma::vec current_param_val_v,
    arma::uvec judge_start_ind,
    arma::uvec case_year_v, arma::uvec case_judge_year_v,
    int alpha_v_1_start_ind, int alpha_v_2_start_ind,
    int delta_v_1_start_ind, int delta_v_2_start_ind,
    arma::uvec pos_judge_ind, arma::uvec pos_judge_year,
    arma::uvec neg_judge_ind, arma::uvec neg_judge_year) {


  for (int i = 0; i < pos_judge_ind.n_elem; i++) {
    if (current_param_val_v(pos_judge_ind(i)) < 0) {
      arma::uvec judge_year = find(case_judge_year_v == pos_judge_year(i));
      arma::uvec cases = find(case_year_v == pos_judge_year(i));
      current_param_val_v(judge_year) =
        -current_param_val_v(judge_year);
      current_param_val_v(alpha_v_1_start_ind + cases) =
        -current_param_val_v(alpha_v_1_start_ind + cases);
      current_param_val_v(alpha_v_2_start_ind + cases) =
        -current_param_val_v(alpha_v_2_start_ind + cases);
      current_param_val_v(delta_v_1_start_ind + cases) =
        -current_param_val_v(delta_v_1_start_ind + cases);
      current_param_val_v(delta_v_2_start_ind + cases) =
        -current_param_val_v(delta_v_2_start_ind + cases);
    }
  }
  for (int i = 0; i < neg_judge_ind.n_elem; i++) {
    if (current_param_val_v(neg_judge_ind(i)) > 0) {
      arma::uvec judge_year = find(case_judge_year_v == neg_judge_year(i));
      arma::uvec cases = find(case_year_v == neg_judge_year(i));
      current_param_val_v(judge_year) =
        -current_param_val_v(judge_year);
      current_param_val_v(alpha_v_1_start_ind + cases) =
        -current_param_val_v(alpha_v_1_start_ind + cases);
      current_param_val_v(alpha_v_2_start_ind + cases) =
        -current_param_val_v(alpha_v_2_start_ind + cases);
      current_param_val_v(delta_v_1_start_ind + cases) =
        -current_param_val_v(delta_v_1_start_ind + cases);
      current_param_val_v(delta_v_2_start_ind + cases) =
        -current_param_val_v(delta_v_2_start_ind + cases);
    }
  }
  return(current_param_val_v);
}

// [[Rcpp::export]]
List sample_three_utility_probit_gp(
    arma::mat vote_m, arma::mat all_param_draws, arma::mat y_star_m_1, arma::mat y_star_m_2, arma::mat y_star_m_3,
    arma::uvec judge_start_inds, arma::uvec judge_end_inds, arma::uvec case_years,
    arma::umat case_judge_years_ind_m, arma::uvec judge_year_v,
    int alpha_v_1_start_ind, int alpha_v_2_start_ind,
    int delta_v_1_start_ind, int delta_v_2_start_ind, int rho_ind,
    arma::vec alpha_mean_v, arma::mat alpha_cov_s, arma::vec delta_mean_v, arma::mat delta_cov_s,
    double rho_mean,double rho_sigma, double rho_sd, int num_iter, int start_iter,
    int keep_iter, arma::uvec pos_judge_ind, arma::uvec neg_judge_ind,
    arma::uvec pos_judge_year, arma::uvec neg_judge_year) {


  arma::vec current_param_val_v = all_param_draws.row(0).t();
  // arma::vec accept_count(zeta_param_start_ind - psi_param_start_ind);
  // accept_count.zeros();
  for (int i = 0; i < num_iter; i++) {
    if (i % 100 == 0) {
      Rcout << i << "\n";
    }

    for (int j = 0; j < vote_m.n_rows; j++) {
      for (int k = 0; k < vote_m.n_cols; k++) {
        if (!is_finite(vote_m(j, k))) {
          continue;
        }
        arma::vec y_star_vec = {y_star_m_1(j, k),
                          y_star_m_2(j, k),
                          y_star_m_3(j, k)};
        arma::vec out_v = sample_y_star_m(
          y_star_vec, vote_m(j, k),
          current_param_val_v(alpha_v_1_start_ind + k),
          current_param_val_v(alpha_v_2_start_ind + k),
          current_param_val_v(judge_start_inds(j) + case_judge_years_ind_m(j, k)),
          current_param_val_v(delta_v_1_start_ind + k),
          current_param_val_v(delta_v_2_start_ind + k));
        y_star_m_1(j, k) = out_v(0);
        y_star_m_2(j, k) = out_v(1);
        y_star_m_3(j, k) = out_v(2);
      }
    }

    for (unsigned int j = 0; j < vote_m.n_rows; j++) {
      arma::uvec current_ind = {j};
      arma::uvec interested_inds = find_finite(vote_m.row(j).t());
      rowvec y_star_m_1_v = y_star_m_1.row(j);
      rowvec y_star_m_3_v = y_star_m_3.row(j);
      arma::uvec judge_years_v = case_judge_years_ind_m.row(j).t();
      current_param_val_v(span(
          judge_start_inds(j), judge_end_inds(j))) =
        sample_three_utility_probit_beta_gp(
          y_star_m_1.submat(current_ind, interested_inds),
          y_star_m_3.submat(current_ind, interested_inds),
          current_param_val_v(alpha_v_1_start_ind + interested_inds).t(),
          current_param_val_v(alpha_v_2_start_ind + interested_inds).t(),
          current_param_val_v(delta_v_1_start_ind + interested_inds).t(),
          current_param_val_v(delta_v_2_start_ind + interested_inds).t(),
          judge_years_v(interested_inds), current_param_val_v(rho_ind));
    }

    arma::vec match_var_v(vote_m.n_cols);
    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      arma::uvec current_ind = {j};
      arma::uvec interested_inds = find_finite(vote_m.col(j));
      arma::vec delta_v = {current_param_val_v(delta_v_1_start_ind + j),
                     current_param_val_v(delta_v_2_start_ind + j)};
      arma::uvec judge_years_v = case_judge_years_ind_m.col(j);
      arma::vec out_v =
        sample_three_utility_probit_matched_alpha(
          y_star_m_1.submat(interested_inds, current_ind),
          y_star_m_3.submat(interested_inds, current_ind),
          current_param_val_v(
            judge_start_inds(interested_inds) +
            judge_years_v(interested_inds)),
          delta_v, alpha_mean_v, alpha_cov_s,
          delta_mean_v, delta_cov_s);

      current_param_val_v(alpha_v_1_start_ind + j) = out_v(0);
      current_param_val_v(alpha_v_2_start_ind + j) = out_v(1);
      match_var_v(j) = out_v(2);
    }

    for (unsigned int j = 0; j < vote_m.n_cols; j++) {
      arma::uvec current_ind = {j};
      arma::uvec interested_inds = find_finite(vote_m.col(j));
      arma::vec alpha_v = {current_param_val_v(alpha_v_1_start_ind + j),
                     current_param_val_v(alpha_v_2_start_ind + j)};
      arma::uvec judge_years_v = case_judge_years_ind_m.col(j);
      arma::vec out_v =
        sample_three_utility_probit_matched_delta(
          y_star_m_1.submat(interested_inds, current_ind),
          y_star_m_3.submat(interested_inds, current_ind),
          alpha_v, current_param_val_v(
              judge_start_inds(interested_inds) +
                judge_years_v(interested_inds)),
          match_var_v(j), delta_mean_v, delta_cov_s);
      current_param_val_v(delta_v_1_start_ind + j) = out_v(0);
      current_param_val_v(delta_v_2_start_ind + j) = out_v(1);
    }

    if (pos_judge_ind.n_elem > 0 || neg_judge_ind.n_elem > 0) {
      current_param_val_v(span(0, rho_ind - 1)) =
        adjust_all_judge_ideology(
          current_param_val_v(span(0, rho_ind - 1)),
          judge_start_inds, case_years, judge_year_v,
          alpha_v_1_start_ind, alpha_v_2_start_ind,
          delta_v_1_start_ind, delta_v_2_start_ind,
          pos_judge_ind, pos_judge_year,
          neg_judge_ind, neg_judge_year);
    }

    current_param_val_v(rho_ind) = sample_rho_pos_logit_gibbs(
      current_param_val_v(rho_ind),
      current_param_val_v(span(0, alpha_v_1_start_ind - 1)),
      judge_start_inds, judge_end_inds, rho_mean, rho_sigma, rho_sd);

    int post_burn_i = i - start_iter + 1;
    if (i >= start_iter && (fmod(post_burn_i, keep_iter) == 0)) {
      int keep_iter_ind = post_burn_i / keep_iter - 1;
      all_param_draws.row(keep_iter_ind) = current_param_val_v.t();
    }
  }

  return(List::create(Named("param_draws") = all_param_draws,
                      Named("y_star_m_1") = y_star_m_1,
                      Named("y_star_m_2") = y_star_m_2,
                      Named("y_star_m_3") = y_star_m_3));
}

double phid(double x) {
  return 0.5 * (1.0 + erf(x / sqrt(2.0)));
}




// [[Rcpp::export]]
arma::mat calc_probit_bggum_three_utility_post_prob_m(
    arma::mat leg_ideology, arma::mat alpha_m, arma::mat delta_m,
    arma::mat case_vote_m, int num_votes) {

  arma::mat post_prob(case_vote_m.n_rows, case_vote_m.n_cols, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    for (int j = 0; j < case_vote_m.n_cols; j++) {
      for (int i = 0; i < case_vote_m.n_rows; i++) {
        double mean_1 =
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j));
        double mean_2 =
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j + 1));
        post_prob(i, j) += bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
      }
    }
  }
  return(post_prob);
}

// [[Rcpp::export]]
arma::vec calc_waic_probit_bggum_three_utility(
  arma::mat leg_ideology, arma::mat alpha_m, arma::mat delta_m,
  arma::mat case_vote_m, int num_votes) {

  arma::vec mean_prob(num_votes, fill::zeros);
  arma::vec mean_log_prob(num_votes, fill::zeros);
  arma::vec log_prob_var(num_votes, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    if (iter + 1 % 100 == 0) {
      Rcout << iter << "\n";
    }
    int vote_num = 0;
    for (int j = 0; j < case_vote_m.n_cols; j++) {
      for (int i = 0; i < case_vote_m.n_rows; i++) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        double mean_1 =
          alpha_m(iter, 2 * j) * (
            leg_ideology(iter, i) - delta_m(iter, 2 * j));
        double mean_2 =
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j + 1));
        double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        double log_prob = case_vote_m(i, j) * log(yea_prob) +
          (1 - case_vote_m(i, j)) * log(1 - yea_prob);
        mean_prob(vote_num) += exp(log_prob);
        double next_mean_log_prob = (iter * mean_log_prob(vote_num) + log_prob) / (iter + 1);
        log_prob_var(vote_num) +=
          (log_prob - mean_log_prob(vote_num)) * (log_prob - next_mean_log_prob);
        mean_log_prob(vote_num) = next_mean_log_prob;
        vote_num++;
      }
    }
    // Rcout << vote_num << endl;
  }
  return(
    log(mean_prob / leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}

// [[Rcpp::export]]
arma::vec calc_waic_probit_bggum_three_utility_block(
    arma::mat leg_ideology, arma::mat alpha_m, arma::mat delta_m,
    arma::mat case_vote_m, arma::uvec case_year, arma::mat block_m) {

  arma::vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  arma::vec mean_log_prob(block_m.n_rows, fill::zeros);
  arma::vec log_prob_var(block_m.n_rows, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    Rcout << iter << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int i = block_m(ind, 0);
      int year = block_m(ind, 1);
      int judge_ind = i + (year - 1) * case_vote_m.n_rows;
      double log_prob = 0;
      arma::uvec interested_cases = find(case_year == year);
      for (int j : interested_cases) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        double mean_1 =
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, judge_ind) - delta_m(iter, 2 * j));
        double mean_2 =
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, judge_ind) - delta_m(iter, 2 * j + 1));
        double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        log_prob += case_vote_m(i, j) * log(yea_prob) +
          (1 - case_vote_m(i, j)) * log(1 - yea_prob);
      }
      mean_prob(ind) = max(mean_prob(ind), log_prob) +
        log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
      double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
      log_prob_var(ind) +=
        (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
      mean_log_prob(ind) = next_mean_log_prob;
    }
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}


// [[Rcpp::export]]
arma::vec calc_waic_probit_bggum_three_utility_block_rcpp(
    arma::mat leg_ideology, arma::mat alpha_m, arma::mat delta_m,
    arma::mat case_vote_m, arma::uvec case_year, arma::mat block_m) {

  arma::vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  arma::vec mean_log_prob(block_m.n_rows, fill::zeros);
  arma::vec log_prob_var(block_m.n_rows, fill::zeros);
  // Rcout << case_vote_m << endl;
  // double corr = 0.5;
  // double sd = sqrt(2);
  // arma::mat lower_cov = {{2, 1},
  //                  {1, 2}};
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    // if (iter + 1 % 100 == 0) {
    //   Rcout << iter << "\n";
    // }
    Rcout << iter << endl;
    // int vote_num = 0;
    // Rcout << vote_num << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int i = block_m(ind, 0);
      int year = block_m(ind, 1);
      // int judge_ind = i + (year - 1) * case_vote_m.n_rows;
      double log_prob = 0;
      arma::uvec interested_cases = find(case_year == year);
      for (int j : interested_cases) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        // int judge_ind = i + (case_year(j) - 1) * case_vote_m.n_rows;
        // Rcout << judge_ind << endl;
        double mean_1 =
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, ind) - delta_m(iter, 2 * j));
        double mean_2 =
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, ind) - delta_m(iter, 2 * j + 1));
        double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        log_prob += case_vote_m(i, j) * log(yea_prob) +
          (1 - case_vote_m(i, j)) * log(1 - yea_prob);
      }
      mean_prob(ind) = max(mean_prob(ind), log_prob) +
        log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
      double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
      log_prob_var(ind) +=
        (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
      mean_log_prob(ind) = next_mean_log_prob;
    }
    // Rcout << vote_num << endl;
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}

// [[Rcpp::export]]
arma::vec calc_waic_probit_bggum_three_utility_block_vote_rcpp(
    arma::mat leg_ideology, arma::mat alpha_m, arma::mat delta_m,
    arma::mat case_vote_m, arma::mat block_m) {

  arma::vec mean_prob(block_m.n_rows);
  mean_prob.fill(-datum::inf);
  arma::vec mean_log_prob(block_m.n_rows, fill::zeros);
  arma::vec log_prob_var(block_m.n_rows, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    Rcout << iter << endl;
    for (int ind = 0; ind < block_m.n_rows; ind++) {
      int j = block_m(ind, 0);
      int year = block_m(ind, 1);
      double log_prob = 0;
      for (int i = 0; i < case_vote_m.n_rows; i++) {
        if (!is_finite(case_vote_m(i, j))) {
          continue;
        }
        double mean_1 =
          alpha_m(iter, 2 * j) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j));
        double mean_2 =
          alpha_m(iter, 2 * j + 1) * (
              leg_ideology(iter, i) - delta_m(iter, 2 * j + 1));
        double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        log_prob += case_vote_m(i, j) * log(yea_prob) +
          (1 - case_vote_m(i, j)) * log(1 - yea_prob);
      }
      mean_prob(ind) = max(mean_prob(ind), log_prob) +
        log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
      double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
      log_prob_var(ind) +=
        (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
      mean_log_prob(ind) = next_mean_log_prob;
    }
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}


// arma::vec calc_waic(
//     arma::mat leg_ideology, arma::mat alpha1_m, arma::mat alpha2_m, arma::mat delta1_m, arma::mat delta2_m,
//     arma::mat case_vote_m) {

//   arma::vec mean_prob(case_vote_m.n_cols);
//   mean_prob.fill(-datum::inf);
//   arma::vec mean_log_prob(case_vote_m.n_cols, fill::zeros);
//   arma::vec log_prob_var(case_vote_m.n_cols, fill::zeros);
//   for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
//     Rcout << iter << endl;
//     for (int ind = 0; ind < case_vote_m.n_cols; ind++) {
//       double log_prob = 0;
//       for (int i = 0; i < case_vote_m.n_rows; i++) {
//         if (!is_finite(case_vote_m(i, ind))) {
//           continue;
//         }
//         double mean_1 =
//           alpha1_m(iter, ind) * (
//               leg_ideology(iter, i) - delta1_m(iter, ind));
//         double mean_2 =
//           alpha2_m(iter, ind) * (
//               leg_ideology(iter, i) - delta2_m(iter, ind));
//         double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
//         yea_prob = min(yea_prob, 1 - 1e-9);
//         yea_prob = max(yea_prob, 1e-9);
//         log_prob += case_vote_m(i, ind) * log(yea_prob) +
//           (1 - case_vote_m(i, ind)) * log(1 - yea_prob);
//       }
//       mean_prob(ind) = max(mean_prob(ind), log_prob) +
//         log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
//       double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
//       log_prob_var(ind) +=
//         (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
//       mean_log_prob(ind) = next_mean_log_prob;
//     }
//   }
//   return(
//     mean_prob - log(leg_ideology.n_rows) -
//       (log_prob_var) / (leg_ideology.n_rows - 1));
// }

// [[Rcpp::export]]
arma::vec calc_waic(
    arma::mat leg_ideology, arma::mat alpha1_m, arma::mat alpha2_m, arma::mat delta1_m, arma::mat delta2_m,
    arma::mat case_vote_m) {

  arma::vec mean_prob(case_vote_m.n_rows);
  mean_prob.fill(-datum::inf);
  arma::vec mean_log_prob(case_vote_m.n_rows, fill::zeros);
  arma::vec log_prob_var(case_vote_m.n_rows, fill::zeros);
  for (int iter = 0; iter < leg_ideology.n_rows; iter++) {
    Rcout << iter << endl;
    for (int ind = 0; ind < case_vote_m.n_rows; ind++) {
      double log_prob = 0;
      for (int j = 0; j < case_vote_m.n_cols; j++) {
        if (!is_finite(case_vote_m(ind, j))) {
          continue;
        }
        double mean_1 =
          alpha1_m(iter, j) * (
              leg_ideology(iter, ind) - delta1_m(iter, j));
        double mean_2 =
          alpha2_m(iter, j) * (
              leg_ideology(iter, ind) - delta2_m(iter, j));
        double yea_prob = bvnd(-mean_1 / sqrt(2), -mean_2 / sqrt(2), 0.5);
        yea_prob = min(yea_prob, 1 - 1e-9);
        yea_prob = max(yea_prob, 1e-9);
        log_prob += case_vote_m(ind, j) * log(yea_prob) +
          (1 - case_vote_m(ind, j)) * log(1 - yea_prob);
      }
      mean_prob(ind) = max(mean_prob(ind), log_prob) +
        log(1 + exp(min(mean_prob(ind), log_prob) - max(mean_prob(ind), log_prob)));
      double next_mean_log_prob = (iter * mean_log_prob(ind) + log_prob) / (iter + 1);
      log_prob_var(ind) +=
        (log_prob - mean_log_prob(ind)) * (log_prob - next_mean_log_prob);
      mean_log_prob(ind) = next_mean_log_prob;
    }
  }
  return(
    mean_prob - log(leg_ideology.n_rows) -
      (log_prob_var) / (leg_ideology.n_rows - 1));
}