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


class JustinDistribution(dist.Distribution):
    """Class for modeling time-to-failure."""
    arg_constraints = {
        "y_0": dist.constraints.nonnegative,
        "t_1": dist.constraints.positive,
        "peak": dist.constraints.nonnegative,
        "t_2": dist.constraints.positive,
        "k_2": dist.constraints.less_than(0),
        "c": dist.constraints.nonnegative
    }
    # We put that the support is positive even though
    # it is really all non-negative real numbers.
    # I don't think it makes a difference.
    support = dist.constraints.positive
    def __init__(self, y_0, k_1, t_1, peak, t_2, k_2, c):
        """
        Args:
            y_0: 0 <= y_0
            k_1: any non-zero real number. k_1*t_1 < 25.
            t_1: t_1 > 0. k_1*t_1 < 25.
            peak: 0 <= peak
            t_2: t_1 <= t_2
            k_2: k_2 < 0
            c: c >= 0
        """
        promoted_shapes = util.promote_shapes(y_0, k_1, t_1, peak, t_2, k_2, c)
        y_0 = promoted_shapes[0]
        k_1 = promoted_shapes[1]
        t_1 = promoted_shapes[2]
        peak = promoted_shapes[3]
        t_2 = promoted_shapes[4] 
        k_2 = promoted_shapes[5] 
        c = promoted_shapes[6]

        # Introduce any other variable that we may want to use later.
        a_1 = (y_0 - peak)/(1-jnp.exp(k_1 * t_1))

        self.a_1 = a_1
        
        # Save others to self.
        self.y_0 = y_0
        self.k_1 = k_1
        self.t_1 = t_1
        self.peak = peak
        self.t_2 = t_2
        self.k_2 = k_2
        self.c = c
        
        super().__init__(batch_shape=jnp.shape(peak), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError
    
    def hazard(self, t):
        """
        See: An Introduction to Survival Analysis 
        Using Stata, Second Edition by Mario Cleves
        
        https://books.google.de/books?id=xttbn0a-QR8C&printsec=frontcover&hl=de#v=onepage&q&f=false
        """
        a_1 = self.a_1
        y_0 = self.y_0
        k_1 = self.k_1
        t_1 = self.t_1
        peak = self.peak
        t_2 = self.t_2 
        k_2 = self.k_2 
        c = self.c

        part_0 = jnp.exp(k_1 * t_1)
        # The hazard function is defined piecewise. 
        # The pieces are below.
        h_1 = a_1 * jnp.exp(k_1 * t) + (peak - y_0 * part_0)/(1 - part_0)
        h_2 = peak
        h_3 = (peak - c)*jnp.exp(k_2 * (t - t_2)) + c

        h_with_12 = jnp.where(
            t > t_1,
            h_2,
            h_1
        )

        h_with_123 = jnp.where(
            t > t_2,
            h_3,
            h_with_12
        )

        return h_with_123
    

    def cum_haz(self, t):
        y_0 = self.y_0
        k_1 = self.k_1
        t_1 = self.t_1
        peak = self.peak
        t_2 = self.t_2 
        k_2 = self.k_2 
        c = self.c

        part_0 = jnp.exp(k_1 * t_1)

        cum_haz_1 = ((y_0 - peak)*(jnp.exp(k_1 * t) - 1) + k_1*(peak - y_0*part_0)*t) / ((1 - part_0)*k_1)

        part_1 = ((y_0 - peak)*(part_0 - 1) + k_1*(peak - y_0*part_0)*t_1) / ((1 - part_0)*k_1)

        cum_haz_2 = part_1 + peak * (t - t_1)

        part_2 = part_1 + peak * (t_2 - t_1)

        cum_haz_3 = part_2 + (peak - c)/k_2*jnp.exp(k_2*(t - t_2)) + c*t - (peak - c)/k_2 - c*t_2

        cum_haz_2_with_1 = jnp.where(
            t > t_1,
            cum_haz_2,
            cum_haz_1
        )

        cum_haz_3_with_21 = jnp.where(
            t > t_2,
            cum_haz_3,
            cum_haz_2_with_1
        )

        return cum_haz_3_with_21
    

    def survival(self, t):
        """P(T > t)"""
        return jnp.exp(-self.cum_haz(t=t))

    def log_prob(self, t):
        h = self.hazard(t=t)
        cum_hazard = self.cum_haz(t=t)

        return jnp.log(h) - cum_hazard
    

class JustinDistribution_2(dist.Distribution):
    """Class for modeling time-to-failure."""
    arg_constraints = {
        
    }
    # We put that the support is positive even though
    # it is really all non-negative real numbers.
    # I don't think it makes a difference.
    support = dist.constraints.nonnegative
    def __init__(self, weights):
        """
        Args:
            weights: len(weights) == 7
        """
        num_gompertz_parts = 5
        num_flatline_parts = 1
        num_lognormal_parts = 1
        num_parts_for_haz = num_gompertz_parts + num_flatline_parts + num_lognormal_parts
        normalized_weights = weights / jnp.sum(weights)
        parts_for_haz = []
        for i in range(num_gompertz_parts):
            multiple = 7*(i + 1)
            parts_for_haz.append(
                dist.Gompertz(
                    concentration=1.0/np.exp(multiple), 
                    rate=1
                )
            )

        parts_for_haz.append(
            {
                "flatline_haz_part": lambda x: jnp.where(
                        x > 7,
                        10**(-6)*x,
                        jnp.array([0])
                    ),
                
                "flatline_cum_haz_part": lambda x: jnp.where(
                        x > 7,
                        10**(-6) * x**2 - (10**(-6) * 7**2),
                        jnp.array([0])
                    )
            }
        ) 

        parts_for_haz.append(
            dist.LogNormal(
                loc=np.log(14),
                scale=1
            )
        )
            
        self.num_gompertz_parts = num_gompertz_parts
        self.parts_for_haz = parts_for_haz
        self.num_parts_for_haz = num_parts_for_haz
        self.weights = normalized_weights
        
        super().__init__(batch_shape=(), event_shape=())

    def sample(self, key, sample_shape=()):
        raise NotImplementedError
    
    def hazard(self, t):
        parts_for_haz = self.parts_for_haz
        num_parts_for_haz = self.num_parts_for_haz
        num_gompertz_parts = self.num_gompertz_parts 
        weights = self.weights 

        unweighted_haz_vals = np.zeros(shape=(num_parts_for_haz, len(t)))
        for i in range(num_gompertz_parts):
            unweighted_haz_vals[i] = jnp.exp(
                parts_for_haz[i].log_prob(value=t)
            )

        unweighted_haz_vals[-2] = parts_for_haz[-2]["flatline_haz_part"](t)

        unweighted_haz_vals[-1] = jnp.exp(
            parts_for_haz[-1].log_prob(value=t)
        )

        return weights @ unweighted_haz_vals

    def cum_haz(self, t):
        parts_for_haz = self.parts_for_haz
        num_parts_for_haz = self.num_parts_for_haz
        num_gompertz_parts = self.num_gompertz_parts 
        weights = self.weights 

        unweighted_cum_haz_vals = np.zeros(shape=(num_parts_for_haz, len(t)))
        for i in range(num_gompertz_parts):
            unweighted_cum_haz_vals[i] = parts_for_haz[i].cdf(value=t)
            
        unweighted_cum_haz_vals[-2] = parts_for_haz[-2]["flatline_haz_part"](t)
        unweighted_cum_haz_vals[-1] = parts_for_haz[-1].cdf(t)
        return weights @ unweighted_cum_haz_vals
    

    def survival(self, t):
        """P(T > t)"""
        return jnp.exp(-self.cum_haz(t=t))

    def log_prob(self, t):
        h = self.hazard(t=t)
        cum_hazard = self.cum_haz(t=t)

        return jnp.log(h) - cum_hazard
    
    def pdf(self, t):
        h = self.hazard(t=t)
        cum_hazard = self.cum_haz(t=t)

        return h * jnp.exp(-cum_hazard)