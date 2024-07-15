#!/usr/bin/env python3
"""The module contains classes and functions useful for decision analysis."""

import numpy as np
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.util as util
import jax.numpy as jnp


def get_sr_weights(n:int) -> np.ndarray:
    """Get the sum-reciprocal surrogate weights for each of
    
    the n attributes.
    Source: https://doi.org/10.1287/deca.2022.0456
    """
    w = np.ones(shape=(n,))

    ranks = np.arange(
        start=1,
        stop=n + 1,
        dtype=np.int64
    )

    reciprocal_ranks = w / ranks
  
    fracs = (n + 1 - ranks)/n

    numerators = fracs + reciprocal_ranks

    denominator = np.sum(
        a=numerators,
        axis=0
    )

    w = numerators / denominator

    return w

# https://realpython.com/python-super/#an-overview-of-pythons-super-function
class Pert(dist.Beta):
    """Class for modified-PERT distribution from numpyro
    with parameters
    a: min
    b: mode
    c: max
    gamma: Lower values of gamma make for a
        distribution that is less peaked 
        at the mode.
        gamma > 0
    """
    def __init__(self, a, b, c, gamma=4):
        # https://pubsonline.informs.org/doi/epdf/10.1287/ited.1080.0013
        # Davis 2008 formula for conversion
        # between PERT and beta distributions
        # https://en.wikipedia.org/wiki/PERT_distribution#The_modified-PERT_distribution
        mu = (a + gamma * b + c)/(gamma + 2)
        # https://reference.wolfram.com/language/ref/PERTDistribution.html
        sigma_squared = (c - a - b*gamma + c*gamma)*(c + b*gamma - a * (1 + gamma))/((2 + gamma)**2 * (3 + gamma))
        alpha_plus_beta = (mu - a)*(c - mu)/sigma_squared - 1 
        alpha = (mu - a)/(c - a)*alpha_plus_beta
        beta = (c - mu)/(c - a)*alpha_plus_beta
        concentration1 = alpha
        concentration0 = beta
        super().__init__(concentration1, concentration0)
    

class JustinDistribution(dist.MixtureGeneral):
    """Class for modeling time-to-failure."""
    arg_constraints = {
        "modes": dist.constraints.greater_than(3)
    }
    support = dist.constraints.nonnegative
    def __init__(self, modes=jnp.arange(5, 30, 2)):
        """
        Args:
            modes: positive integers greater than 3

        """
        support = dist.constraints.nonnegative
        component_dists = []
    
        # Create all of the Chi-Squared distributions.
        for mode in modes:
            df = mode + 2
            component_dists.append(
                dist.Chi2(df=df)
            )
        # Weight the Chi-Squared distributions.
        num_chisqs = len(modes)
        chisq_weights = jnp.ones(shape=(num_chisqs,))
        chisq_rel_weight = 0.5

        component_dists.append(
                dist.Exponential(rate=1.0/30.0)
            )
        exp_rel_weight = jnp.array([0.5])
        probs = jnp.append(
            chisq_rel_weight*chisq_weights/num_chisqs,
            exp_rel_weight
        )
   
        mixture_proportions = dist.Categorical(
            probs=probs
        )

        super().__init__(
            mixing_distribution=mixture_proportions, 
            component_distributions=component_dists,
            support=support
        )








       
    
 