data {
  int<lower=0> J;                       // number of students/legislators
  int<lower=0> K;                       // number of items/bills
  int<lower=0> D;                       // number of dimensions
  int<lower=0> N;                       // number of observations
  array[N] int<lower=0,upper=K> kk;     // item/bill for observation n
  array[N] int<lower=0,upper=J> jj;     // student for observation n
  array[N] int<lower=0,upper=1>  y;     // score/vote for observation n
}
transformed data {
  vector[D] theta_mean;
  vector[D] theta_scale;
  
  for (d in 1:D) {
    theta_mean[d] = 0;
    theta_scale[d] = 1;
  }

  vector[D] alpha_mean;
  vector[D] alpha_scale;
  
  for (d in 1:D) {
    alpha_mean[d] = 0;
    alpha_scale[d] = 1;
  }
}
parameters {
  matrix[J, D] theta;                   // student ability/  ideal points (J legislators x D dimensions)
  corr_matrix[D] theta_corr;            // theta correlation matrix
  
  matrix[K, D] alpha;                   // item/bill discriminations
  corr_matrix[D] alpha_corr;            // Independent discrimination parameters

  vector[K] beta;                       // item/bill difficulty
}

model {
  // Priors
  theta_corr ~ lkj_corr(1);
  for (j in 1:J) {
    theta[j] ~ multi_normal(theta_mean, quad_form_diag(theta_corr, theta_scale));
  }
  // alpha_corr ~ lkj_corr(0.05); // let stan find it implicitly
  for (k in 1:K) {
    alpha[k] ~ multi_normal(alpha_mean, quad_form_diag(alpha_corr, alpha_scale));
  }

  beta ~ normal(0, 10);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(dot_product(theta[jj[n]], alpha[kk[n]]) - beta[kk[n]]);
  }
}
