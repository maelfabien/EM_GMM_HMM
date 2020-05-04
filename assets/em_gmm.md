Let us now run EM on GMMs. In the E-step, we will estimate the value of an "auxiliary function", which is in fact a lower boud of the gain in likelihood that we generate by updating the value of the parameters. In the M-step, we maximize this auxiliary function with respect to the parameters of the GMM, and obtain updated parameters. 

These new parameters are fed into the E-step again, and so on... We stop if:
- a stopping criteria is met (e.g. marginal gain is below a given threshold)
- we reach the maximum number of iterations

In the animation below, slide the cursors of the number of iterations and the number of components of the GMM, and observe the contour plots of the GMM displaying the mean and variance of the components of the GMM. Note that there is a theoretical guarantee that the likelihood increases with the EM algorithm, and we illustrate it by displaying the likelihood in terms of the number of iterations below.