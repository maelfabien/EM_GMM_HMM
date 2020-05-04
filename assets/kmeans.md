k-Means algorithm is in fact a sepcial case of GMM/EM. It is what is called a hard EM (we only keep in mind the cluster with the highest probability). It classifies data based on their mean, and not looking at the covariance (or assuming an identity covariance matrix in the GMM). Therefore, when clusters overlap for example, k-Means typically has some issues and does not perform well.

In the illustration below, we notice how the GMM approach outperforms the k-Means. However, k-Means is often used as a simple approach to intialize the parameters of the EM/GMM:
- we take the means identified by the k-Means clusters
- we compute the intra-cluster variance 
- we compute the proportion of points allocated to each cluster as the weights of the Gaussians