# Multidimensional Item Response Theory Model

This report presents a Bayesian multidimensional Item Response Theory (MIRT) model for analyzing binary response data across multiple dimensions, with key insights on correlation structure modeling.

## Model Specification

### Data Structure
- $J$ individuals (students/legislators)
- $K$ items (questions/bills)  
- $D$ latent dimensions
- $N$ total observations
- $y_{jk} \in \{0,1\}$ binary response of individual $j$ to item $k$

### Parameters

**Individual Ability Parameters:**
$\boldsymbol{\theta}_j \in \mathbb{R}^D \quad \text{for } j = 1,\ldots,J$

**Item Discrimination Parameters:**
$\boldsymbol{\alpha}_k \in \mathbb{R}^D \quad \text{for } k = 1,\ldots,K$

**Item Difficulty Parameters:**
$\beta_k \in \mathbb{R} \quad \text{for } k = 1,\ldots,K$

**Correlation Structures:**
These correlations, represent how different dimensions of ideology correlate across legislators. This matches real-world expectations (e.g., economic and social conservatism are often correlated):

- $\boldsymbol{\Omega}_\theta$ : $D \times D$ correlation matrix for abilities
- $\boldsymbol{\Omega}_\alpha$ : $D \times D$ correlation matrix for discriminations (with improper prior)

### Likelihood Function

$$P(y_{jk} = 1) = \text{logit}^{-1}\left(\boldsymbol{\theta}_j \cdot \boldsymbol{\alpha}_k - \beta_k\right)$$

$$y_{jk} \sim \text{Bernoulli}\left(\text{logit}^{-1}\left(\sum_{d=1}^D \theta_{jd} \alpha_{kd} - \beta_k\right)\right)$$

### Prior Distributions

**Ability Parameters (Proper Multivariate Prior):**
$\boldsymbol{\Omega}_\theta \sim \text{LKJ}(1)$
$\boldsymbol{\theta}_j \sim \mathcal{N}_D(\boldsymbol{0}_D, \boldsymbol{\Omega}_\theta) \quad \text{for } j = 1,\ldots,J$

The LKJ distribution is used for $\boldsymbol{\Omega}_\theta$ because it provides a principled prior over the space of correlation matrices. The LKJ(η) distribution has density:
$p(\boldsymbol{\Omega}) \propto |\boldsymbol{\Omega}|^{\eta-1}$

where $|\boldsymbol{\Omega}|$ is the determinant. For $\eta = 1$, this yields a uniform distribution over correlation matrices, ensuring no a priori bias toward any particular correlation structure while maintaining proper normalization. The LKJ distribution also has favorable computational properties, generating correlation matrices that are guaranteed to be positive definite and avoiding the boundary issues that can arise with other parameterizations.

**Item Discrimination Parameters (Correlated with Improper Prior):**
$\boldsymbol{\alpha}_k \sim \mathcal{N}_D(\boldsymbol{0}_D, \boldsymbol{\Omega}_\alpha) \quad \text{for } k = 1,\ldots,K$

where $\boldsymbol{\Omega}_\alpha$ has an **improper uniform prior** over the space of valid correlation matrices.

**Item Difficulty Parameters:**
$\beta_k \sim \mathcal{N}(0, 10) \quad \text{for } k = 1,\ldots,K$

## Correlation Results With DW Nominate
**Substantal Improvement** was achieved in terms of correlation with the dimensions identified by DW Nominate over both Dimensions.

### Current Result:

| Dimension   | Correlation | P-value  |
|-------------|-------------|----------|
| Dimension 1 | 0.981490    | 0.000000 |
| Dimension 2 | -0.566155   | 0.000000 |


### Old Result:
| Dimension   | Correlation | P-value  |
|-------------|-------------|----------|
| Dimension 1 | -0.848313   | 0.000000 |
| Dimension 2 | -0.108809   | 0.276328 |


Current results are comparable to the correlation with DW Nominate found using the IDEAL package in R:

### R-IDEAL:
| Dimension   | Correlation | P-value  |
|-------------|-------------|----------|
| Dimension 1 | 0.982336    | 0.000000 |
| Dimension 2 | 0.680133    | 0.000000 |

## Key Modeling Insights

### Improper vs Proper Priors for Correlation Matrices

**Critical Finding:** For the discrimination correlation matrix $\boldsymbol{\Omega}_\alpha$, using an improper uniform prior (by declaring the parameter without specifying a prior) yields substantially better correlation with DW NOminate in the **2nd Dimension** compared to proper LKJ priors.

**Comparison of Approaches:**
- **No correlation structure** (independent $\alpha$ parameters): ~10% correlation with DW-NOMINATE
- **LKJ(0.1) prior**: ~40% correlation with DW-NOMINATE  
- **Improper uniform prior**: ~56% correlation with DW-NOMINATE

**(Possible) Explanation:** Legislative voting often has clear dimensional structure. We want to recover the "true" correlations from data, not impose prior beliefs. I think, DW-NOMINATE also doesn't impose correlation constraints on item parameters. The improper uniform prior $p(\boldsymbol{\Omega}_\alpha) \propto 1$ allows STAN to find the best likelihood under the hood i.e., the correlation structure to be determined entirely by the likelihood without Bayesian shrinkage. This approximates maximum likelihood estimation for the correlation parameters:

$$\hat{\boldsymbol{\Omega}}_\alpha = \arg\max_{\boldsymbol{\Omega}_\alpha} \mathcal{L}(\boldsymbol{\Omega}_\alpha | \text{data})$$

In contrast, proper LKJ priors impose regularization that pulls correlations toward zero (LKJ(1)) or extreme values (LKJ(η < 1)), reducing the model's ability to recover the true dimensional structure present in legislative voting data.

### Asymmetric Treatment of Correlation Structures

The model employs **asymmetric priors** for the two correlation matrices:
- **$\boldsymbol{\Omega}_\theta$**: Proper LKJ(1) prior for substantive interpretability
- **$\boldsymbol{\Omega}_\alpha$**: Improper uniform prior for optimal empirical fit

This reflects their different roles: ability correlations represent meaningful relationships between ideological dimensions, while discrimination correlations are primarily technical parameters for capturing item behavior patterns.

### Computational Considerations

**Advantages of Improper Prior:**
- Maximum likelihood-like estimation of correlation structure
- Superior recovery of true dimensional relationships
- Better alignment with established measures (e.g., DW-NOMINATE)

**Possible Potential Risks according to Claude:**
- Improper posterior distributions
- Possible convergence issues
- Overfitting without regularization

**Mitigation:** Monitor convergence diagnostics ($\hat{R} < 1.01$, effective sample sizes, trace plot mixing) to ensure computational stability despite the improper prior.

### Model Identification and Parameterization

The difficulty parameterization $-\beta_k$ follows classical test theory conventions where larger $\beta_k$ values correspond to more difficult items. The compensatory multidimensional structure allows high ability in any dimension to compensate for deficiencies in others, weighted by each item's discrimination pattern $\boldsymbol{\alpha}_k$.