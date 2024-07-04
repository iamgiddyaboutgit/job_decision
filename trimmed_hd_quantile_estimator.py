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
from scipy import stats, optimize
import numpy as np

def estimate_quantiles(x:np.ndarray, probs:float|tuple|np.ndarray, width=None) -> np.ndarray:
    """Use the Trimmed Harrell-Davis quantile estimator to estimate
    
    quantiles from sample data for a continous population 
    distribution.
    
    Args:
        x: sample data
        probs: real numbers between 0 and 1 (exclusive)
        width: Default: `1/sqrt(len(x))`. It is required that
            `0 < width <= 1`.

    Returns:
        Estimated quantiles
    """
    #########################################################
    # Perform Data Validation
    #########################################################
    if probs is None:
        return None
    # Make sure probs is a numpy.ndarray.
    probs = np.array([probs]).ravel()
    num_quantiles_to_get = len(probs)
    
    # Make sure x is a vector.
    x = x.ravel()
    # Determine the desired truncated beta distributions to use.
    # Determine effective sample size.
    # Without weights, this is just the number of data points.
    n = len(x)
    if len(x) < 2:
        # There's not enough information to estimate 
        # quantiles effectively.
        estimated_quantiles = np.repeat(
            a=x,
            repeats=num_quantiles_to_get
        )
        return estimated_quantiles
    
    #########################################################
    # Perform Main Logic
    #########################################################
    # Note that a different truncated beta distribution
    # will be used for each entry in probs.
    alphas = get_alpha_param(n=n, p=probs)
    betas = get_beta_param(n=n, p=probs)
    
    # Sort x so that we can easily identify the order statistics.
    x.sort()

    if width is None:
        # Use the suggested default.
        width = 1/np.sqrt(n)

    # Fill estimated_quantiles in loop.
    estimated_quantiles = np.zeros(shape=(num_quantiles_to_get,))

    for i in range(num_quantiles_to_get):
        # For each desired quantile, we have
        # to go through this long process of weighting
        # the order statistics using weights
        # derived from a custom truncated beta distribution.
       
        alpha = alphas[i]
        beta = betas[i]
        hdi = get_beta_hdi(
            alpha=alpha,
            beta=beta,
            width=width
        )
        # seq_to_eval contains points at which to evaluate
        # the custom truncated beta CDF according to the
        # formula in the article.
        seq_to_eval = np.linspace(start=1/n, stop=1, num=n, endpoint=True)

        cdf_vals = truncated_beta_cdf(
            x=seq_to_eval,
            alpha=alpha,
            beta=beta,
            lower=hdi[0],
            upper=hdi[1]
        )

        # Do some fancy indexing to subtract cdf_vals
        # from a shifted version of itself.
        order_stat_weights = np.zeros(shape=(n,))
        order_stat_weights[0] = cdf_vals[0]
        # https://stackoverflow.com/a/69306263
        # Assign values into order_stat_weights
        # from the 1-index onwards.
        if n > 1:
            order_stat_weights[1:] = cdf_vals[1:] - cdf_vals[:-1]
        
        estimated_quantiles[i] = np.inner(order_stat_weights, x)

    return estimated_quantiles


def truncated_beta_cdf(x, alpha, beta, lower, upper):
    """Return the value of the truncated Beta Cumulative Distribution
    
    Function at x.  The beta distribution is truncated at both
    ends as determined by lower and upper.  
    
    Args:
        x: array_like. Vector of quantiles. These can be any 
            real numbers.
        alpha: concentration parameter. alpha > 0.
        beta: concentration parameter. beta > 0.
        lower: lower bound for truncated support of distribution.
        upper: upper bound for truncated support of distribution.
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
        (x < lower),
        lower,
        x
    )
    x_2 = np.where(
        (x_1 > upper),
        upper,
        x_1
    )

    beta_cdf_lower = beta_dist.cdf(x=lower)
    beta_cdf_upper = beta_dist.cdf(x=upper)
    beta_cdf_x = beta_dist.cdf(x=x_2)
    trunc_beta_cdf_vals = (beta_cdf_x - beta_cdf_lower)/(beta_cdf_upper - beta_cdf_lower)

    return trunc_beta_cdf_vals

def get_alpha_param(n, p):
    """Get the alpha parameter to use for the
    Harrell-Davis quantile estimator.

    Args:
        n: Effective sample size.
        p: Quantiles.
    """
    return (n + 1)*np.array(p)

def get_beta_param(n, p):
    """Get the beta parameter to use for the
    Harrell-Davis quantile estimator.

    Args:
        n: Effective sample size.
        p: Quantiles.
    """
    return (n + 1)*np.array(1 - p)

def get_beta_mode(alpha, beta):
    """Get the mode of a Beta distribution
    
    with parameters `alpha` and `beta`.
    See: https://en.wikipedia.org/wiki/Beta_distribution

    For the case where `alpha == beta`, return None
    even though technically the mode is any value
    in the interval (0, 1).

    Returns:
        tuple of mode value (if applicable) and
            mode type where the mode type is
            0: mode of 0
            1: mode of 1
            2: bimodal
            3: central mode
            4: weird mode (i.e. mode is any value in (0, 1))

    """
    if alpha > 1 and beta > 1:
        # unimodal
        mode = (alpha - 1)/(alpha + beta - 2)
        mode_type = 3
    elif alpha < 1 and beta > 1:
        mode = 0
        mode_type = 0
    elif alpha > 1 and beta < 1:
        mode = 1
        mode_type = 1
    elif alpha < 1 and beta < 1:
        # bimodal
        mode = {0, 1}
        mode_type = 2
    else:
        mode = None
        mode_type = 4

    return (mode, mode_type)

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
            0 < width <= 1
    """
    hdi = np.zeros(shape=(2,), dtype=np.float64)
    mode, mode_type = get_beta_mode(alpha=alpha, beta=beta)

    def for_root_finding(x, alpha, beta, width):
        beta_dist = stats.beta(
            a=alpha,
            b=beta
        )

        return beta_dist.pdf(x) - beta_dist.pdf(x + width)

    # Handle an extreme case first.
    if width == 1:
        hdi[0] = 0
        hdi[1] = 1
    elif mode_type == 3:
        # Perform numerical optimization to determine where
        # the left border of our HDI is.
        # lower_bracket for the root:
        lower_bracket = max(0, mode - width)
        # upper_bracket for the root:
        upper_bracket = min(mode, 1 - width)
        hdi[0] = optimize.toms748(
            f=for_root_finding,
            a=lower_bracket,
            b=upper_bracket,
            args=(alpha, beta, width),
            full_output=False,
            disp=True
        )
        hdi[1] = hdi[0] + width
    elif mode_type == 0:
        hdi[0] = 0
        hdi[1] = width
    elif mode_type == 1:
        hdi[0] = 1 - width
        hdi[1] = 1
    elif mode_type == 2:
        # bimodal
        raise RuntimeError("Bimodal distribution.")
    elif mode_type == 4:
        # weird mode
        raise RuntimeError("HDI is not identifiable.")
    
    return hdi

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


    