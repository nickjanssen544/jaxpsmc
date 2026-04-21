from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax

from .tools_jax import *
from .student_jax import *






@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Geometry:
    """
    Class stores fitted geometry parameters for two distributions.

    Parameters:
    -----------
    normal_mean: 
        mean vector for normal fit, shape (D,).
    normal_cov: 
        covariance matrix for normal fit, shape (D, D).
    t_mean: 
        mean vector for Student-t fit, shape (D,).
    t_cov: 
        covariance matrix for Student-t fit, shape (D, D).
    t_nu: 
        degrees of freedom for Student-t fit.

    Returns:
    -------
    Geometry object:
        stores fitted normal and Student-t parameters 
        so the code can pass them around as one JAX pytree.
    """    
    normal_mean: jax.Array  # (D,)
    normal_cov:  jax.Array  # (D,D)
    t_mean:      jax.Array  # (D,)
    t_cov:       jax.Array  # (D,D)
    t_nu:        jax.Array  # ()

    def tree_flatten(self):
        """
        Function converts object into JAX pytree children.

        Parameters:
        -----------
        None:
            it uses current Geometry object.

        Returns:
        ---------
        tuple:
            with the stored arrays and auxiliary data.
        """
        return (self.normal_mean, self.normal_cov, self.t_mean, self.t_cov, self.t_nu), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Function rebuilds Geometry object from pytree children.

        Parameters:
        -----------
        aux: 
            auxiliary pytree data. It is unused here.
        children: 
            tuple with the stored array fields.

        Returns:
        --------
        new Geometry object.
        """ 
        # unpack saved fields    
        nm, nc, tm, tc, tnu = children
        # rebuild dataclass
        return cls(nm, nc, tm, tc, tnu)

    @classmethod
    def init(cls, dim: int, *, dtype=jnp.float64):
        """
        Function creates an empty Geometry object for a given dimension.

        Parameters:
        -----------
        dim: 
            dimension of the parameter space.
        dtype: 
            numeric dtype used for arrays.

        Returns:
        --------
        Geometry object:
            with zero means, zero covariances, and large t_nu.
        """
        # create zero mean vectors      
        z1 = jnp.zeros((dim,), dtype=dtype)
        # create zero covariance matrice
        z2 = jnp.zeros((dim, dim), dtype=dtype)
        # use large value so the initial t distribution is close to normal
        nu = jnp.asarray(1e6, dtype=dtype)
        # return initialized object
        return cls(z1, z2, z1, z2, nu)


@jax.jit
def _cov_unweighted(theta: jax.Array, *, jitter: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Function computes the unweighted sample mean and covariance.

    Parameters:
    -----------
    theta: 
        sample matrix with shape (N, D).
    jitter: 
        small value added to the diagonal for numerical stability.

    Output:
    -------
    mu: 
        sample mean with shape (D,).
    cov: 
        sample covariance with shape (D, D).
    """
    # convert input to JAX array
    theta = jnp.asarray(theta)
    # read sample count and dimension
    n, d = theta.shape
    # compute sample mean
    mu = jnp.mean(theta, axis=0)
    # center samples around mean
    xc = theta - mu[None, :]
    # use n - 1 for sample covariance denominator
    denom = jnp.asarray(n - 1, theta.dtype)
    # avoid division by zero when n <= 1
    cov = (xc.T @ xc) / jnp.where(denom > 0, denom, jnp.asarray(1.0, theta.dtype))
    # force matrix to be symmetric
    cov = 0.5 * (cov + cov.T)
    # add diagonal jitter for stability
    cov = cov + jitter * jnp.eye(d, dtype=theta.dtype)
    return mu, cov


@jax.jit
def _cov_weighted_aweights(theta: jax.Array, weights: jax.Array, *, jitter: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Function computes weighted mean and covariance.

    Parameters:
    -----------
    theta: 
        sample matrix with shape (N, D).
    weights: 
        nonnegative sample weights with shape (N,).
    jitter: 
        small value added to the diagonal for numerical stability.

    Returns:
    --------
    mu: 
        weighted mean with shape (D,).
    cov: 
        weighted covariance with shape (D, D).
    """
    # convert inputs to JAX arrays
    theta = jnp.asarray(theta)
    w = jnp.asarray(weights)
    # read the sample count and dimension
    n, d = theta.shape
    dtype = theta.dtype

    # validate weights 
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)

    # normalize weights if they valid
    w = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # calculate weighted mean
    mu = jnp.sum(theta * w[:, None], axis=0)
    # center samples around weighted mean.
    xc = theta - mu[None, :]

    # correction factor for normalized analytical weights.
    # normalization: fact = 1 / (1 - sum(w^2))   because w is normalized to sum=1
    w2sum = jnp.sum(w * w)
    denom = (jnp.asarray(1.0, dtype) - w2sum)
    fact = jnp.where(denom > 0, jnp.asarray(1.0, dtype) / denom, jnp.asarray(0.0, dtype))

    # compute weighted covariance
    cov = (xc * w[:, None]).T @ xc
    cov = cov * fact
    # force matrix to be symmetric
    cov = 0.5 * (cov + cov.T)
    # diagonal jitter gives stability here 
    cov = cov + jitter * jnp.eye(d, dtype=dtype)

    # go back to unweighted result if the weights are bad
    mu_u, cov_u = _cov_unweighted(theta, jitter=jitter)
    mu = jnp.where(bad, mu_u, mu)
    cov = jnp.where(bad, cov_u, cov)
    return mu, cov


@partial(jax.jit, static_argnames=("nu_cap",))
def _sanitize_nu(nu: jax.Array, nu_cap: float) -> jax.Array:
    """
    Function replaces non-finite nu values with a fixed cap.

    Parameters:
    -----------
    nu: 
        degrees of freedom value.
    nu_cap: 
        replacement value used when nu is not finite.

    Returns:
    --------
        finite nu value
    """
    # convert cap to same dtype as nu 
    cap = jnp.asarray(nu_cap, dtype=nu.dtype)
    # keep nu if finite, otherwise use cap
    return jnp.where(jnp.isfinite(nu), nu, cap)


@partial(jax.jit, static_argnames=("nu_cap",))
def geometry_fit_jax(
    geom: Geometry,
    theta: jax.Array,          # (N,D)
    weights: jax.Array,        # (N,)
    use_weights: jax.Array,    # bool scalar: if True, use weights logic
    key: jax.Array,            # PRNGKey
    *,
    nu_cap: float = 1e6,
    jitter: float = 1e-9,
):
    """
    Function fits Geometry object from sample points.

    Parameters:
    -----------
    geom: 
        current Geometry object. It is not used directly for fit, but
        it keeps function interface consistent.
    theta: 
        sample matrix with shape (N, D).
    weights: 
        sample weights with shape (N,).
    use_weights: 
        boolean flag. If True, use weighted logic
    key: 
        JAX random key used for resampling.
    nu_cap: 
        upper fallback value used when t_nu is not finite.
    jitter: 
        small value added to covariance diagonals.

    Returns:
    --------
    geom_new: 
        fitted Geometry object.
    key_out: 
        output random key.
    resample_status: 
        status code from the resampling step.
    """
    # convert to jax
    theta = jnp.asarray(theta)
    weights = jnp.asarray(weights)
    use_weights = jnp.asarray(use_weights, dtype=bool)
    # match jitter dtype to sample array
    jitter = jnp.asarray(jitter, dtype=theta.dtype)

    def _do_weighted(_):
        """
        Function computes weighted normal statistics.

        Parameters:
        -----------
        _: 
            unused input required by lax.cond.

        Returns:
            weighted mean and covariance.
        """
        # use weighted covariance 
        return _cov_weighted_aweights(theta, weights, jitter=jitter)

    def _do_unweighted(_):
        """
        Function computes unweighted normal statistics.

        Parameters:
        -----------
        _: 
            unused input required by lax.cond.

        Returns:
            unweighted mean and covariance.
        """
        # use unweighted covariance
        return _cov_unweighted(theta, jitter=jitter)
    # choose weighted or unweighted normal fit
    normal_mean, normal_cov = lax.cond(use_weights, _do_weighted, _do_unweighted, operand=None)

    # read nr of samples
    n = theta.shape[0]

    def _t_fit_resampled(_):
        """
        Function fits Student-t geometry after resampling.

        Parameters:
        -----------
        _: 
            unused input required by lax.cond.

        Returns:
        t_mean: 
            fitted Student-t mean.
        t_cov: 
            fitted Student-t covariance.
        t_nu: 
            fitted degrees of freedom.
        key_out: 
            updated random key.
        status: 
            resampling status code.
        """
        # normalize weights for resampling
        wsum = jnp.sum(weights)
        bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(weights)) | jnp.any(weights < 0)
        w_norm = weights / jnp.where(bad, jnp.asarray(1.0, theta.dtype), wsum)

        # resample indices using systematic resampling
        idx, status, key_out = systematic_resample_jax(w_norm, key=key)
        # keep indices inside valid bounds
        idx_safe = jnp.clip(idx, 0, n - 1)
        # build resampled sample matrix
        theta_rs = theta[idx_safe]
        # fit multivariate Student-t model on resampled data
        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta_rs)  
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key_out, status

    def _t_fit_direct(_):
        """
        Function fits the Student-t geometry without resampling.

        Parameters:
        -----------
        _: 
            unused input required by lax.cond.

        Returns:
        --------
        t_mean: 
            fitted Student-t mean.
        t_cov: 
            fitted Student-t covariance.
        t_nu: 
            fitted and cleaned degrees of freedom.
        key: 
            unchanged random key.
        status: 
            zero status code.
        """
        # fit multivariate Student-t model directly
        t_mean, t_cov, t_nu, _info = fit_mvstud_jax(theta)
        # use zero to mark that no resampling error occurred
        status = jnp.int64(0)
        return t_mean, t_cov, _sanitize_nu(t_nu, nu_cap), key, status
    # choose direct fit or resampled fit for the Student-t model
    t_mean, t_cov, t_nu, key_out, resample_status = lax.cond(use_weights, _t_fit_resampled, _t_fit_direct, operand=None)
    # build new Geometry object from the fitted values
    geom_new = Geometry(
        normal_mean=normal_mean,
        normal_cov=normal_cov,
        t_mean=t_mean,
        t_cov=t_cov,
        t_nu=t_nu,
    )
    return geom_new, key_out, resample_status



