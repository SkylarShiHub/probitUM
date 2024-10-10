\name{post_samples}
\alias{post_samples}
\title{Posterior Sample Examples of Parameters}
\description{
  A list of posterior samples generated with the function 'sample_three_utility_probit'. For each parameter, there are 1000 iterations.
}
\usage{
  data(post_samples)
}
\format{
  A list containing the following elements:
    \itemize{
      \item \code{beta}: A matrix of beta parameter samples.
      \item \code{alpha1}: A matrix of alpha1 parameter samples.
      \item \code{alpha2}: A matrix of alpha2 parameter samples.
      \item \code{delta1}: A matrix of delta1 parameter samples.
      \item \code{delta2}: A matrix of delta2 parameter samples.
    }
}
\source{
  Simulated posterior samples based on the dataset 'house_votes_m'.
}
\keyword{datasets}