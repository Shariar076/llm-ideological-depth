data {
  int<lower=1> J;                     // number of legislators
  int<lower=1> K;                     // number of bills
  int<lower=1> N;                     // number of observations
  int<lower=1> D;                     // number of dimensions (2 for politics)
  array[N] int<lower=1, upper=J> jj;  // legislator for observation n
  array[N] int<lower=1, upper=K> kk;  // bill for observation n
  array[N] int<lower=0, upper=1> y;   // vote for observation n (1=Yea, 0=Nay)
  
  // Identification constraints (optional - can fix specific legislators)
  int<lower=0> n_fixed;               // number of legislators with fixed positions
  array[n_fixed] int<lower=1, upper=J> fixed_legs;  // which legislators to fix
  matrix[n_fixed, D] fixed_pos;       // their fixed positions
}

parameters {
  // Ideal points for non-fixed legislators
  matrix[J - n_fixed, D] alpha_free;  // ideal points (free parameters)
  
  // Bill parameters
  matrix[K, D] beta;                  // bill discrimination vectors (direction)
  vector[K] alpha_bill;               // bill difficulty parameters
  
  // Scale parameters
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
}

transformed parameters {
  // Combine fixed and free ideal points
  matrix[J, D] alpha;
  
  if (n_fixed > 0) {
    int free_idx = 1;
    for (j in 1:J) {
      int is_fixed = 0;
      for (f in 1:n_fixed) {
        if (j == fixed_legs[f]) {
          alpha[j] = fixed_pos[f];
          is_fixed = 1;
          break;
        }
      }
      if (!is_fixed) {
        alpha[j] = alpha_free[free_idx];
        free_idx += 1;
      }
    }
  } else {
    // Alternative: standardize ideal points (mean 0, sd 1 per dimension)
    alpha = alpha_free;
    for (d in 1:D) {
      real mean_d = mean(alpha_free[, d]);
      real sd_d = sd(alpha_free[, d]);
      alpha[, d] = (alpha_free[, d] - mean_d) / sd_d;
    }
  }
}

model {
  // Priors
  // Ideal points prior
  for (i in 1:(J - n_fixed)) {
    alpha_free[i] ~ normal(0, sigma_alpha);
  }
  
  // Bill parameters priors
  for (k in 1:K) {
    beta[k] ~ normal(0, sigma_beta);
  }
  alpha_bill ~ normal(0, 1);
  
  // Hyperpriors
  sigma_alpha ~ cauchy(0, 2.5);
  sigma_beta ~ cauchy(0, 2.5);
  
  // Likelihood - following equation (2) from the paper
  // P(y_ij = 1) = Φ(b'_j * x_i - α_j)
  for (n in 1:N) {
    real utility_diff = dot_product(beta[kk[n]], alpha[jj[n]]) - alpha_bill[kk[n]];
    y[n] ~ bernoulli(Phi(utility_diff));
  }
}

generated quantities {
  // Posterior predictive checks
  array[N] int y_rep;
  vector[N] log_lik;
  
  for (n in 1:N) {
    real utility_diff = dot_product(beta[kk[n]], alpha[jj[n]]) - alpha_bill[kk[n]];
    y_rep[n] = bernoulli_rng(Phi(utility_diff));
    log_lik[n] = bernoulli_lpmf(y[n] | Phi(utility_diff));
  }
}