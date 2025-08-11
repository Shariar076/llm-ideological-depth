data {
  int<lower=0> J;                       // number of students/legislators
  int<lower=0> K;                       // number of items/bills
  int<lower=0> D;                       // number of dimensions
  int<lower=0> N;                       // number of observations
  array[N] int<lower=0,upper=K> kk;     // item/bill for observation n
  array[N] int<lower=0,upper=J> jj;     // student for observation n
  array[N] int<lower=0,upper=1>  y;     // score/vote for observation n
  
  // Reference legislators for identification
  int<lower=1,upper=J> ref_j1;          // Most liberal legislator
  int<lower=1,upper=J> ref_j2;          // Most conservative legislator  
  int<lower=1,upper=J> ref_j3;          // Update: othogonal to 1st two
  // int<lower=1,upper=J> ref_j3;          // Extreme negative on dim2;
  // int<lower=1,upper=J> ref_j4;          // Extreme positive on dim2
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
  matrix[J, D] theta_raw;               // unconstrained ideal points
  corr_matrix[D] theta_corr;            // theta correlation matrix
  
  matrix[K, D] alpha;                   // item/bill discriminations
  corr_matrix[D] alpha_corr;            // Independent discrimination parameters

  vector[K] beta;                       // item/bill difficulty
}

transformed parameters {
  matrix[J, D] theta;                   // constrained ideal points
  
  // Copy unconstrained parameters
  theta = theta_raw;
  
  // Apply identification constraints
  theta[ref_j1, 1] = -2.0;              // Most liberal: strongly negative on dim1
  theta[ref_j2, 1] = 2.0;               // Most conservative: strongly positive on dim1
  theta[ref_j3, 1] = 0.0;               // fixes rotation
  theta[ref_j3, 2] = 2.0;               // High on dim2 (or -2.0 if negative extreme)
  // theta[ref_j3, 2] = -1.5;              // Extreme negative on dim2
  // theta[ref_j4, 2] = 1.5;               // Extreme positive on dim2

}

model {
  // Priors
  theta_corr ~ lkj_corr(0.5);
  
  // Prior for unconstrained legislators
  for (j in 1:J) {
    if (j != ref_j1 && j != ref_j2 && j != ref_j3) { // && j != ref_j4
      theta_raw[j] ~ multi_normal(theta_mean, quad_form_diag(theta_corr, theta_scale));
    }
  }
  
  // Priors for reference legislators (only for unconstrained dimensions)
  theta_raw[ref_j1, 2] ~ normal(0, 1);  // ref_j1 dim2 is free
  theta_raw[ref_j2, 2] ~ normal(0, 1);  // ref_j2 dim2 is free
  theta_raw[ref_j3, 1] ~ normal(0, 1);  // ref_j3 dim1 is free  
  // theta_raw[ref_j4, 1] ~ normal(0, 1);  // ref_j4 dim1 is free

  for (k in 1:K) {
    alpha[k] ~ multi_normal(alpha_mean, quad_form_diag(alpha_corr, alpha_scale));
  }

  beta ~ normal(0, 10);
  
  // Likelihood
  for (n in 1:N) {
    y[n] ~ bernoulli_logit(dot_product(theta[jj[n]], alpha[kk[n]]) - beta[kk[n]]);
  }
}