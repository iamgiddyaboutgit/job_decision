#!/usr/bin/env python3
"""Estimate quantiles of a continuous distribution from sample data.

Python code is adapted from the R code and ideas in:
Article: Trimmed Harrell-Davis quantile estimator based on the 
    highest density interval of the given width
Author: Andrey Akinshin
URL: https://doi.org/10.1080/03610918.2022.2050396
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


    