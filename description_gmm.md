A GMM is a weighted sum of M components Gaussian densities.A density of a Gaussian can be defined as:

$$ P(x \mid \lambda) = \sum_{k=1}^M w_k \times g(x \mid \mu_k, \Sigma_k) $$

Where:
- $ x $ is a D-dimensional feature vector
- $ w_k, k = 1, 2, ..., M $ is the mixture weights
- $ \mu_k, k = 1, 2, ..., M $ is mean of each Gaussian
- $ \Sigma_k, k = 1, 2, ..., M $ is the covariance of each Gaussian
- $ g(x \mid \mu_k, \Sigma_k) $ are the Gaussian densities such that:

$$ g(x \mid \mu_k, \Sigma_k) = \frac{1}{(2 \pi)^{\frac{D}{2}} {\mid \Sigma_k \mid}^{\frac{1}{2}}} exp^{ - \frac{1}{2}(x - \mu_k)^T \Sigma_k^{-1} (x-\mu_k)} $$

The parameters of the GMM are therefore : $ \lambda = (w_k, \mu_k, \Sigma_k), k = 1, 2, 3, ..., M $. How do we solve GMM? Using Expectation-Maximization (EM) algorithm. EM algorithm is an interative algorithm which, at each steap, tries to update the parameters in order to maximize the likelihood, and eventually finds the maximum likelihood. Formally, EM algorithm iterative update of the parameters should lead to :

$ \prod_{n=1}^N P(x_n \mid \lambda) $ 

$ \geq \prod_{n=1}^N P(x_n \mid \lambda^{old}) $

The maximization problem of the EM algorithm is in fact:

$$ Q(\lambda, \lambda^{old}) = \sum_{n=1}^N $$

$$ \sum_{k = 1}^M P(k \mid x_n) \log w_k g(x_n \mid \mu_k, \sigma_k) $$

Where :

$$ P(k \mid x) = \frac{w_k g(x \mid \mu_k, \Sigma_k)}{P(x \mid \lambda^{old})} $$

The starting parameters are usually identified by k-means approach.
