#!/usr/bin/env python3
"""Estimate quantiles of a continuous distribution from sample data.

Python code is adapted from the R code and ideas in:
Article: Trimmed Harrell-Davis quantile estimator based on the 
    highest density interval of the given width
Author: Andrey Akinshin
URL: https://doi.org/10.1080/03610918.2022.2050396

and

Akinshin, A. (2023). Weighted quantile estimators. arXiv preprint arXiv:2304.07265.
URL: https://arxiv.org/pdf/2304.07265
"""
from scipy import stats
import numpy as np

def truncated_beta_cdf(x, alpha, beta, l, r):
    """Return the value of the truncated Beta Cumulative Distribution
    
    Function at x.  The beta distribution is truncated at both
    ends as determined by l and r.  
    
    Args:
        x: array_like. Vector of quantiles. These can be any 
            real numbers.
        alpha: concentration parameter. alpha > 0.
        beta: concentration parameter. beta > 0.
        l: lower bound for truncated support of distribution.
        r: upper bound for truncated support of distribution.
    """
    # Initialize parent Beta distribution.
    beta_dist = stats.beta(
        a=alpha,
        b=beta
    )

    # Handle values of x to the left or right
    # of the truncated support by changing them
    # so that they are properly evaluated by 
    # the parent Beta CDF.
    # Note that we use l and r here because
    # the truncated Beta CDF function at these points
    # is 0 and 1, respectively (as desired).
    # See: https://en.wikipedia.org/wiki/Truncated_distribution
    x_1 = np.where(
        (x < l),
        l,
        x
    )
    x_2 = np.where(
        (x_1 > r),
        r,
        x_1
    )

    beta_cdf_l = beta_dist.cdf(x=l)
    beta_cdf_r = beta_dist.cdf(x=r)
    beta_cdf_x = beta_dist.cdf(x=x_2)
    trunc_beta_cdf_vals = (beta_cdf_x - beta_cdf_l)/(beta_cdf_r - beta_cdf_l)

    return trunc_beta_cdf_vals

def get_alpha_param(n, p):
    """Get the alpha parameter to use for the
    Harrell-Davis quantile estimator.

    Args:
        n: Effective sample size.
        p: Quantiles.
    """
    return (n + 1)*p

def get_beta_param(n, p):
    """Get the beta parameter to use for the
    Harrell-Davis quantile estimator.

    Args:
        n: Effective sample size.
        p: Quantiles.
    """
    return (n + 1)*(1 - p)

def beta_mode(alpha, beta):
    """Get the mode of a Beta distribution
    
    with parameters alpha and beta.
    See: https://en.wikipedia.org/wiki/Beta_distribution

    For the case where alpha == beta, return None
    even though technically the mode is any value
    in the interval (0, 1).
    """
    if alpha > 1 and beta > 1:
        # unimodal
        mode = (alpha - 1)/(alpha + beta - 2)
    elif alpha < 1 and beta < 1:
        # bimodal
        mode = {0, 1}
    elif alpha < 1 and beta > 1:
        mode = 0
    elif alpha > 1 and beta < 1:
        mode = 1
    else:
        mode = None

    return mode

def get_beta_hdi(alpha, beta, width):
    """Get the Highest Density Interval (HDI) from
    
    a Beta distribution with parameters `alpha` and `beta`.
    'The HDI is the interval which contains the required 
    mass such that all points within the interval
    have a higher probability density than points outside 
    the interval.' 
    See: https://cran.r-project.org/web/packages/HDInterval/HDInterval.pdf
    Our implementation is a little different from the usual
    because we do not specify the probability mass, but
    rather the desired width of the HDI.

    Args:
        alpha: concentration parameter. alpha > 0.
        beta: concentration parameter. beta > 0.
        width: D in paper 'Trimmed Harrell-Davis . . .'
            0 <= width <= 1
    """


##############################
# getBetaHdi <- function(a, b, width) {
#     eps <- 1e-9
#     if (a < 1 + eps & b < 1 + eps) # Degenerate case  
#         return(c(NA, NA))
#     if (a < 1 + eps & b > 1) # Left border case  
#         return(c(0, width)) 
#     if (a > 1 & b < 1 + eps) # Right border case  
#         return(c(1 - width, 1)) 
#     if (width > 1 - eps)  
#         return(c(0, 1))  # Middle case 
#     mode <- (a - 1)/(a + b - 2) 
    
#     pdf <- function(x) dbeta(x, a, b) 
#     l <- uniroot(f = function(x) pdf(x) - pdf(x + width),  lower = max(0, mode - width),  upper = min(mode, 1 - width),  tol = 1e–9 )$root
#     r <- l + width
#     return(c(l, r))
# }

# thdquantile <- function(x, probs, width = 1/sqrt(length(x)))        
#     sapply(probs, function(p) { 
#                                n <- length(x) 
#         if (n == 0) return(NA) 
#         if (n == 1) return(x)
#         x <- sort(x) 
#         a <- (n + 1) * p 
#         b <- (n + 1) * (1 - p) 
#         hdi <- getBetaHdi(a, b, width)
#         hdiCdf <- pbeta(hdi, a, b)
        
# cdf <- function(xs) {  
#     xs[xs < = hdi[1]] <- hdi[1]
#     xs[xs > = hdi[2]] <- hdi[2]
#     (pbeta(xs, a, b) - hdiCdf[1])/(hdiCdf[2] - hdiCdf[1]) } 
# 
# iL <- floor(hdi[1] * n) 
# iR <- ceiling(hdi[2] * n)  
# cdfs <- cdf(iL:iR/n) 
# W <- tail(cdfs, -1) - head(cdfs, -1) 
# sum(x[(iL + 1):iR] * W)})


    