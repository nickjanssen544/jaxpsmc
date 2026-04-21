from __future__ import annotations

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.special import digamma
jax.config.update("jax_enable_x64", True)
from .bisect_jax import *



#####################################################################
# Nu UPDATE-HELPER
#####################################################################

def _nu_fixed_point_objective(nu: jnp.ndarray, delta: jnp.ndarray, dim: jnp.ndarray) -> jnp.ndarray:
    """
    Function computes the scalar fixed-point objective used to update nu.

    Parameters:
    -----------
        nu: scalar degrees-of-freedom value.
        delta: Mahalanobis distance values with shape (n,).
        dim: data dimension stored as a scalar.

    Returns:
    --------
        scalar objective value whose root defines the updated nu.
    """
    # build Student-t weights for current nu
    w = (nu + dim) / (nu + delta)  # shape (n,)

    # return fixed point objective value
    return (
        -digamma(nu / 2)
        + jnp.log(nu / 2)
        + jnp.mean(jnp.log(w))
        - jnp.mean(w)
        + 1.0
        + digamma((nu + dim) / 2)
        - jnp.log((nu + dim) / 2)
    )


def _opt_nu_bisect(
    delta: jnp.ndarray,
    dim: int,
    nu_old: jnp.ndarray,
    *,
    xtol: jnp.ndarray,
    bisect_maxiter: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.int64, jnp.bool_]:
    """
    Function updates nu by solving the fixed-point objective with bisection.

    Parameters:
    -----------
        delta: Mahalanobis distance values with shape (n,).
        dim: data dimension.
        nu_old: previous nu value.
        xtol: absolute stopping tolerance for bisection.
        bisect_maxiter: maximum number of bisection iterations.

    Returns:
    --------
        updated nu, bisection status code, and boolean flag that marks nu as effectively infinite.
    """
    dtype = delta.dtype
    dim_f = jnp.asarray(dim, dtype=dtype)

    # use very small and very large positive bracket for nu
    a = jnp.asarray(1e-300, dtype=dtype)
    b = jnp.asarray(1e300, dtype=dtype)

    # check large-nu side first
    f_large = _nu_fixed_point_objective(b, delta, dim_f)

    def _set_inf(_: Any):
        """
        Function returns the large-nu shortcut result.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            large nu value, success status, and True flag for infinite nu.
        """
        # treat very large upper bound as infinite nu case
        return (b, jnp.int64(0), jnp.bool_(True))


    def _do_bisect(_: Any):
        """
        Function runs bisection on the fixed-point objective.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            updated nu, status code, and False flag for infinite nu.
        """
        # solve fixed point equation with bisection
        root, status, _, _ = bisect_jax(
            _nu_fixed_point_objective,
            a,
            b,
            xtol=xtol,
            maxiter=bisect_maxiter,
            args=(delta, dim_f),
        )
        # keep the previous nu when bisection fails
        nu_new = jnp.where(status == 0, root, nu_old)
        #return (nu_new, status, jnp.bool_(False))
        return (nu_new, status.astype(jnp.int64), jnp.bool_(False))

    # large-nu shortcut when objective is already nonnegative there
    return lax.cond(f_large >= 0, _set_inf, _do_bisect, operand=None)








#####################################################################
# INITIALIZATION HELPER
#####################################################################
def _init_mu_sigma(data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Function builds the initial location and covariance estimates.

    Parameters:
    -----------
        data: input data array with shape (n, dim).

    Returns:
    --------
        initial mean vector and covariance matrix.
    """
    # sample count and dimension
    n, dim = data.shape
    # coordinate wise median as initial location
    mu = jnp.median(data, axis=0)  

    # stable covariance like starting matrix
    centered = data - jnp.mean(data, axis=0, keepdims=True)
    n_f = jnp.asarray(n, dtype=data.dtype)
    # equivalent to: cov*(n-1)/n + (1/n)*diag(var)
    cov_mle = (centered.T @ centered) / n_f
    var = jnp.var(data, axis=0)  # ddof=0
    Sigma = cov_mle + jnp.diag(var) / n_f

    #Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma








#####################################################################
# EM (EXPECTATION MAXIMIZATION CORE)
#####################################################################

@jax.jit
def _fit_mvstud_core(
    data: jnp.ndarray,              # (n, dim)
    tol: jnp.ndarray,               # scalar float
    max_iter: jnp.ndarray,          # scalar int32
    nu_init: jnp.ndarray,           # scalar float
    xtol: jnp.ndarray,              # scalar float
    bisect_maxiter: jnp.ndarray,    # scalar int32
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.int64, jnp.int64]:
    """
    Function runs the EM updates for a multivariate Student-t fit.

    status:
      0  is converged (|nu - last_nu| <= tol)
      1  is max_iter reached (no convergence)
      2  is nu set to +inf (early-stop condition which is to be specified)
      -k is propagated bisect_jax error codes    

    Parameters:
    -----------
        data: input data array with shape (n, dim).
        tol: stopping tolerance for nu.
        max_iter: maximum number of EM iterations.
        nu_init: initial nu value.
        xtol: absolute stopping tolerance for the nu bisection step.
        bisect_maxiter: maximum number of iterations for the nu bisection step.

    Returns:
    --------
        fitted mean: mu (dim,)
        fitted covariance: Sigma (dim,dim)
        fitted nu: nu (scalar)
        iteration count: iters (int32)
        status code: status (int32)
    """
    n, dim = data.shape
    dtype = data.dtype

    # initial parameters
    mu0, Sigma0 = _init_mu_sigma(data)
    nu0 = nu_init.astype(dtype)
    last_nu0 = jnp.asarray(0.0, dtype=dtype)

    # initial loop state
    i0 = jnp.int64(0)
    stop0 = jnp.bool_(False)
    status0 = jnp.int64(0)

    def cond_fun(state):
        """
        Function checks whether the EM loop should continue.

        Parameters:
        -----------
            state: current EM parameters, counters, and status flags.

        Returns:
        --------
            boolean value that is True when another EM iteration is needed.
        """
        # current loop state
        mu, Sigma, nu, last_nu, i, stop, status = state
        not_done = jnp.logical_and(i < max_iter, jnp.logical_not(stop))
        not_converged = jnp.abs(nu - last_nu) > tol
        return jnp.logical_and(not_done, not_converged)

    def body_fun(state):
        """
        Function performs one EM update.

        Parameters:
        -----------
            state: current EM parameters, counters, and status flags.

        Returns:
        --------
            updated EM loop state.
        """
        # current loop state
        mu, Sigma, nu, last_nu, i, stop, status = state

        # compute Mahalanobis distances under current parameters
        diffs = data - mu[None, :]                    # (n, dim)
        sol = jnp.linalg.solve(Sigma, diffs.T)        # (dim, n)
        delta = jnp.sum(diffs.T * sol, axis=0)        # (n,)

        # update nu with bisection helper
        nu_old = nu
        nu_new, nu_bisect_status, nu_is_inf = _opt_nu_bisect(
            delta, dim, nu_old, xtol=xtol, bisect_maxiter=bisect_maxiter
        )

        # failures in nu update (if exists)
        bisect_error = (nu_bisect_status != 0)

        # compute Student-t weights for the updated nu
        dim_f = jnp.asarray(dim, dtype=dtype)
        w = (nu_new + dim_f) / (nu_new + delta)       # (n,)

        def _keep_params(_: Any):
            """
            Function keeps the current mean and covariance unchanged.

            Parameters:
            -----------
                _: unused operand.

            Returns:
            --------
                current mean and covariance.
            """
            return (mu, Sigma)

        def _update_params(_: Any):
            """
            Function updates the mean and covariance from the current weights.

            Parameters:
            -----------
                _: unused operand.

            Returns:
            --------
                updated mean and covariance.
            """
            # weighted mean
            w_sum = jnp.sum(w)
            mu_upd = jnp.sum(w[:, None] * data, axis=0) / w_sum

            # weighted covariance
            diffs2 = data - mu_upd[None, :]
            Sigma_upd = (diffs2.T * w[None, :]) @ diffs2 / jnp.asarray(n, dtype=dtype)
            Sigma_upd = 0.5 * (Sigma_upd + Sigma_upd.T)
            return (mu_upd, Sigma_upd)

        # match original behavior: if nu becomes inf, return *current* mu/Sigma (don’t update them)
        mu_new2, Sigma_new2 = lax.cond(nu_is_inf, _keep_params, _update_params, operand=None)

        # update stop flag and status code
        stop2 = jnp.logical_or(stop, jnp.logical_or(nu_is_inf, bisect_error))
        status2 = lax.cond(
            status != 0,
            lambda _: status,                                   # already has an error code
            lambda _: lax.cond(
                bisect_error,
                lambda __: nu_bisect_status,                    # negative error code
                lambda __: lax.cond(nu_is_inf, lambda ___: jnp.int64(2), lambda ___: jnp.int64(0), None),
                None,
            ),
            operand=None,
        )

        return (mu_new2, Sigma_new2, nu_new, nu_old, i + jnp.int64(1), stop2, status2)

    # run EM loop
    mu, Sigma, nu, last_nu, iters, stop, status = lax.while_loop(
        cond_fun, body_fun, (mu0, Sigma0, nu0, last_nu0, i0, stop0, status0)
    )

    # decide if convergence is in max_iter
    converged = jnp.abs(nu - last_nu) <= tol
    status = lax.cond(
        status != 0,
        lambda _: status,
        lambda _: lax.cond(converged, lambda __: jnp.int64(0), lambda __: jnp.int64(1), None),
        operand=None,
    )

    return mu, Sigma, nu, iters, status








#####################################################################
# WRAPPER
#####################################################################
def fit_mvstud_jax(
    data,
    tolerance: float = 1e-6,
    max_iter: int = 100,
    nu_init: float = 20.0,
    xtol: float = 2e-12,
    bisect_maxiter: int = 100,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
    """
    Function fits a multivariate Student-t distribution with the EM routine.

    Parameters:
    -----------
        data: input data array with shape (n, dim).
        tolerance: stopping tolerance for nu.
        max_iter: maximum number of EM iterations.
        nu_init: initial nu value.
        xtol: absolute stopping tolerance for the nu bisection step.
        bisect_maxiter: maximum number of iterations for the nu bisection step.

    Returns:
    --------
        fitted mean, fitted covariance, fitted nu, and info dictionary.
    """
    data = jnp.asarray(data)
    tol = jnp.asarray(tolerance, dtype=data.dtype)
    max_iter_j = jnp.asarray(max_iter, dtype=jnp.int64)
    nu_init_j = jnp.asarray(nu_init, dtype=data.dtype)
    xtol_j = jnp.asarray(xtol, dtype=data.dtype)
    bisect_maxiter_j = jnp.asarray(bisect_maxiter, dtype=jnp.int64)

    # run EM core
    mu, Sigma, nu, iters, status = _fit_mvstud_core(
        data, tol, max_iter_j, nu_init_j, xtol_j, bisect_maxiter_j
    )

    # info dictionary returned to caller
    info = {"iters": iters, "status": status}
    return mu, Sigma, nu, info

