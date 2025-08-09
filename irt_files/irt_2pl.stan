data {
  int<lower=1> J;                     // number of legislators
  int<lower=1> K;                     // number of bills
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of dimensions (2 for politics)
  array[N] int<lower=1, upper=J> jj;  // legislator for observation n
  array[N] int<lower=1, upper=K> kk;  // bill for observation n
  array[N] int<lower=0, upper=1> y;   // vote for observation n
}

parameters {
  matrix[J, D] theta;         // ideal points (J legislators x D dimensions)
  corr_matrix[D] theta_corr;  // theta correlation matrix
  matrix[K, D] alpha;         // bill positions
  corr_matrix[D] alpha_corr;  // alpha correlation matrix
  vector[K] beta;             // bill difficulty
  real<lower=0> sigma_theta;
  real<lower=0> sigma_alpha;
}

model {
  // Priors
  theta_corr ~ lkj_corr(1);
  alpha_corr ~ lkj_corr(1);
  for (j in 1:J) theta[j] ~ normal(0, sigma_theta);
  for (k in 1:K) alpha[k] ~ normal(0, sigma_alpha);
  sigma_alpha ~ cauchy(0, 1);
  sigma_alpha ~ cauchy(0, 1);
  beta ~ normal(0, 1);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(dot_product(theta[jj[n]], alpha[kk[n]]) - beta[kk[n]]);
  }
}
