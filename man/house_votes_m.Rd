\name{house_votes_m}
\alias{house_votes_m}
\title{Voting Data from the 116th U.S. House of Representatives}
\description{
  A dataset containing the processed vote data of members in the 116th U.S. House of Representatives on different issues.
}
\usage{
data(house_votes_m)
}
\format{
  A matrix with 429 rows (members) and 907 columns (issues), where:
  \itemize{
    \item 1 indicates a 'Yes' vote,
    \item 0 indicates a 'No' vote,
    \item NA indicates that the member did not vote.
  }
}
\source{
  Jeffrey B. Lewis, Keith Poole, Howard Rosenthal, Adam Boche, Aaron Rudkin, and Luke Sonnet.
  Voteview: Congressional roll-call votes database. https://voteview.com/, 2024. Accessed: 2024-07-15.
}
\keyword{datasets}