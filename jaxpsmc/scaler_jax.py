from __future__ import annotations

from typing import Any, Tuple
from typing import Mapping
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.experimental import checkify
from .input_validation_jax import (
    assert_array_ndim,
    assert_array_2d,
    assert_array_1d,
    assert_arrays_equal_shape,
    assert_equal_type,
    assert_array_float,
    within_interval_mask,
    assert_array_within_interval,
    jit_with_checks,
)



_EMPTY_I32 = jnp.zeros((0,), dtype=jnp.int64)
_DEFAULT_BOUNDS = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float64)  # matches bounds


def init_bounds_config_jax(
    n_dim: int,
    bounds: Array = _DEFAULT_BOUNDS,          # allowed shapes: (2,) or (n_dim, 2)
    periodic: Array = _EMPTY_I32,             # indices, shape (k,)
    reflective: Array = _EMPTY_I32,           # indices, shape (m,)
    *,
    transform: str = "probit",                # "logit" or "probit" 
    scale: bool = True,
    diagonal: bool = True,
) -> dict[str, Array]:
    """
    Function builds the initial scaler configuration from bounds and boundary rules.

    Parameters:
    -----------
        n_dim: number of dimensions.
        bounds: bounds with shape (2,) or (n_dim, 2).
        periodic: indices of periodic dimensions.
        reflective: indices of reflective dimensions.
        transform: bounded transform name, either "logit" or "probit".
        scale: if True, enable affine scaling after the bounds transform.
        diagonal: if True, use diagonal scaling instead of full covariance scaling.

    Returns:
    --------
        configuration dictionary with bounds, masks, and scaling placeholders.
    """
    # n_dim is static because it defines output shapes
    checkify.check(jnp.asarray(n_dim > 0), "n_dim must be a positive integer.")
    n_dim32 = jnp.asarray(n_dim, dtype=jnp.int64)

    # validate bounds array
    bounds = jnp.asarray(bounds, dtype=jnp.float64)
    bounds = assert_array_float(bounds, name="bounds")

    # accept one shared bound pair or one pair per dimension
    ok_bounds_shape = jnp.asarray(
        ((bounds.ndim == 1) & (bounds.shape == (2,))) |
        ((bounds.ndim == 2) & (bounds.shape == (n_dim, 2)))
    )
    checkify.check(ok_bounds_shape, "bounds must have shape (2,) or (n_dim, 2).")

    # distribute shared bounds to all dimensions
    bounds = jnp.broadcast_to(bounds, (n_dim, 2))
    low = bounds[:, 0]
    high = bounds[:, 1]
    checkify.check(jnp.all(low <= high), "bounds[:,0] must be <= bounds[:,1] elementwise.")

    # convert periodic and reflective indices to flat integer arrays
    periodic = jnp.asarray(periodic, dtype=jnp.int64).reshape((-1,))
    reflective = jnp.asarray(reflective, dtype=jnp.int64).reshape((-1,))

    # check that all boundary indices are valid
    checkify.check(jnp.all((periodic >= 0) & (periodic < n_dim)), "periodic indices must be in [0, n_dim).")
    checkify.check(jnp.all((reflective >= 0) & (reflective < n_dim)), "reflective indices must be in [0, n_dim).")

    # convert index lists into boolean masks
    dims = jnp.arange(n_dim, dtype=jnp.int64)
    periodic_mask = jnp.any(dims[:, None] == periodic[None, :], axis=1)
    reflective_mask = jnp.any(dims[:, None] == reflective[None, :], axis=1)

    # dimension cannot be both periodic and reflective
    checkify.check(jnp.all(~(periodic_mask & reflective_mask)),
                   "A dimension cannot be both periodic and reflective.")

    # transform: make it as binary logit=0, probit=1
    is_logit = transform == "logit"     
    is_probit = transform == "probit"
    checkify.check(jnp.asarray(is_logit | is_probit), "transform must be 'logit' or 'probit'.")
    transform_id = jnp.asarray(is_probit, dtype=jnp.int64)

    # placeholder values for parameters learned later by fit_jax
    dtype = bounds.dtype
    nan_vec = jnp.full((n_dim,), jnp.nan, dtype=dtype)
    nan_mat = jnp.full((n_dim, n_dim), jnp.nan, dtype=dtype)
    nan_scalar = jnp.asarray(jnp.nan, dtype=dtype)

    return {
        "ndim": n_dim32,
        "low": low,
        "high": high,
        "periodic_mask": periodic_mask,
        "reflective_mask": reflective_mask,
        "transform_id": transform_id,
        "scale": jnp.asarray(scale),
        "diagonal": jnp.asarray(diagonal),
        "mu": nan_vec,
        "sigma": nan_vec,
        "cov": nan_mat,
        "L": nan_mat,
        "L_inv": nan_mat,
        "log_det_L": nan_scalar,
    }


def masks_jax(low: Array, high: Array) -> dict[str, Array]:
    """
    Function builds bound masks from lower and upper bound arrays.

    Parameters:
    -----------
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).

    Returns:
    --------
        dictionary with masks for left-bounded, right-bounded, both-bounded, and unbounded dimensions.
    """
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # mark finite lower and upper bounds
    fin_low = jnp.isfinite(low)
    fin_high = jnp.isfinite(high)

    # define four bound masks
    mask_none = (~fin_low) & (~fin_high)
    mask_right = (~fin_low) & (fin_high)
    mask_left = (fin_low) & (~fin_high)
    mask_both = (fin_low) & (fin_high)

    return {
        "mask_left": mask_left,
        "mask_right": mask_right,
        "mask_both": mask_both,
        "mask_none": mask_none,
    }


def _create_masks_jax(n_dim: int, bounds: Array) -> dict[str, Array]:
    """
    Function builds masks directly from the dimension count and bounds array.

    Parameters:
    -----------
        n_dim: number of dimensions.
        bounds: bounds with shape (2,) or (n_dim, 2).

    Returns:
    --------
        dictionary with bound masks.
    """
    cfg = init_bounds_config_jax(n_dim, bounds)
    return masks_jax(cfg["low"], cfg["high"])


def _inverse_none_jax(u: Array, mask_none: Array) -> tuple[Array, Array]:
    """
    Function applies the inverse transform on unbounded dimensions.

    Parameters:
    -----------
        u: unconstrained array with shape (N, D).
        mask_none: mask for unbounded dimensions with shape (D,).

    Returns:
    --------
        selected x values and zero log-Jacobian terms.
    """
    u = jnp.asarray(u)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    # unbounded dimensions pass through unchanged
    x = u[:, mask_none]
    log_det_J = jnp.zeros_like(u)[:, mask_none]
    return x, log_det_J


def _forward_none_jax(x: Array, mask_none: Array) -> Array:
    """
    Function applies the forward transform on unbounded dimensions.

    Parameters:
    -----------
        x: constrained array with shape (N, D).
        mask_none: mask for unbounded dimensions with shape (D,).

    Returns:
    --------
        selected u values for the unbounded dimensions.
    """
    x = jnp.asarray(x)
    mask_none = jnp.asarray(mask_none, dtype=bool)

    return x[:, mask_none]


def _inverse_both_jax(
    u: Array,
    low: Array,
    high: Array,
    mask_both: Array,
    transform_id: Array,   # 0 = logit, 1 = probit
) -> tuple[Array, Array]:
    """
    Function applies the inverse transform on dimensions with both finite bounds.

    Parameters:
    -----------
        u: unconstrained array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        mask_both: mask for two-sided bounded dimensions.
        transform_id: integer code, 0 for logit and 1 for probit.

    Returns:
    --------
        transformed x values and diagonal log-Jacobian terms for bounded dimensions.
    """
    # inputs to arrays
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    # choose only bounded dimensions
    u_sel = u[:, mask_both]        # (N, K)
    low_sel = low[mask_both]       # (K,)
    high_sel = high[mask_both]     # (K,)
    span = high_sel - low_sel      # (K,)
    log_span = jnp.log(span)       # (K,)  (invalid bounds will yield nan/-inf)

    def _logit_branch(op):
        """
        Function applies the inverse logit-based bounds transform.

        Parameters:
        -----------
            op: tuple with selected u values and bound data.

        Returns:
        --------
            transformed x values and log-Jacobian terms.
        """
        # define selected arrays
        u_s, low_s, log_span_s, span_s = op

        # map probabilities from sigmoid values into bounded interval
        p = jax.nn.sigmoid(u_s)
        x = p * span_s + low_s
        J = log_span_s + jnp.log(p) + jnp.log1p(-p)
        return x, J

    def _probit_branch(op):
        """
        Function applies the inverse probit-based bounds transform.

        Parameters:
        -----------
            op: tuple with selected u values and bound data.

        Returns:
        --------
            transformed x values and log-Jacobian terms.
        """
        # define selected arrays
        u_s, low_s, log_span_s, span_s = op

        # map Gaussian CDF values into bounded interval
        p = jsp.special.ndtr(u_s)  # Phi(u)
        x = p * span_s + low_s
        # log phi(u) = -0.5 u^2 - log(sqrt(2*pi))
        J = log_span_s + (-0.5 * u_s**2) - jnp.log(jnp.sqrt(2.0 * jnp.pi))
        return x, J

    # choose requested two-sided transform
    x, J = jax.lax.switch(
        transform_id,
        (_logit_branch, _probit_branch),
        (u_sel, low_sel, log_span, span),
    )
    return x, J



def _forward_both_jax(
    x: Array,
    low: Array,
    high: Array,
    mask_both: Array,
    transform_id: Array,   # 0 = logit, 1 = probit
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Function applies the forward transform on dimensions with both finite bounds.

    Parameters:
    -----------
        x: constrained array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        mask_both: mask for two-sided bounded dimensions.
        transform_id: integer code, 0 for logit and 1 for probit.
        eps: clipping value used for numerical stability.

    Returns:
    --------
        transformed u values for bounded dimensions.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    mask_both = jnp.asarray(mask_both, dtype=bool)
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)

    # select only bounded dimensions
    x_sel = x[:, mask_both]          # (N, K)
    low_sel = low[mask_both]         # (K,)
    high_sel = high[mask_both]       # (K,)
    span = high_sel - low_sel        # (K,)

    # convert bounded values to probabilities and clip them away from 0 and 1
    p = (x_sel - low_sel) / span
    eps_t = jnp.asarray(eps, dtype=x_sel.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    def _logit_branch(p_in: Array) -> Array:
        """
        Function applies the logit transform to probabilities.

        Parameters:
        -----------
            p_in: probability array.

        Returns:
        --------
            logit-transformed values.
        """
        # apply logit transform: logit(p) = log(p) - log(1-p) 
        return jnp.log(p_in) - jnp.log1p(-p_in)

    def _probit_branch(p_in: Array) -> Array:
        """
        Function applies the probit transform to probabilities.

        Parameters:
        -----------
            p_in: probability array.

        Returns:
        --------
            probit-transformed values.
        """
        # apply inverse Gaussian CDF: probit(p) = sqrt(2) * erfinv(2p - 1)   
        return jnp.sqrt(jnp.asarray(2.0, dtype=p_in.dtype)) * jsp.special.erfinv(
            2.0 * p_in - 1.0
        )

    # choose requested two-sided transform
    u = jax.lax.switch(transform_id, (_logit_branch, _probit_branch), p)
    return u


def _inverse_right_jax(u: Array, high: Array, mask_right: Array) -> tuple[Array, Array]:
    """
    Function applies the inverse transform on dimensions with only an upper bound.

    Parameters:
    -----------
        u: unconstrained array with shape (N, D).
        high: upper bounds with shape (D,).
        mask_right: mask for right-bounded dimensions.

    Returns:
    --------
        transformed x values and diagonal log-Jacobian terms.
    """
    u = jnp.asarray(u)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    # choose only right-bounded dimensions
    u_sel = u[:, mask_right]        # (N, K)
    high_sel = high[mask_right]     # (K,)

    # apply x = high - exp(u) and keep u as log-Jacobian term
    x = high_sel - jnp.exp(u_sel)   # (N, K) through  broadcasting  
    J = u_sel                       # 
    return x, J


def _forward_right_jax(x: Array, high: Array, mask_right: Array) -> Array:
    """
    Function applies the forward transform on dimensions with only an upper bound.

    Parameters:
    -----------
        x: constrained array with shape (N, D).
        high: upper bounds with shape (D,).
        mask_right: mask for right-bounded dimensions.

    Returns:
    --------
        transformed u values for right-bounded dimensions.
    """
    x = jnp.asarray(x)
    high = jnp.asarray(high)
    mask_right = jnp.asarray(mask_right, dtype=bool)

    # select only right-bounded dimensions
    x_sel = x[:, mask_right]        # (N, K)
    high_sel = high[mask_right]     # (K,)

    # apply u = log(high - x)
    return jnp.log(high_sel - x_sel)



def _inverse_left_jax(u: Array, low: Array, mask_left: Array) -> tuple[Array, Array]:
    """
    Function applies the inverse transform on dimensions with only a lower bound.
    p = exp(u[:, mask_left])
    return exp(u[:, mask_left]) + low[mask_left], u[:, mask_left]
    
    Parameters:
    -----------
        u: unconstrained array with shape (N, D).
        low: lower bounds with shape (D,).
        mask_left: mask for left-bounded dimensions.

    Returns:
    --------
        transformed x values and diagonal log-Jacobian terms.
        x : Array, shape (N, K)
        J : Array, shape (N, K)   (matches original: u[:, mask_left])
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    # select only left-bounded dimensions
    u_sel = u[:, mask_left]      # (N, K)
    low_sel = low[mask_left]     # (K,)

    # apply x = exp(u) + low and keep u as the log-Jacobian term
    x = jnp.exp(u_sel) + low_sel
    J = u_sel                    
    return x, J


def _forward_left_jax(x: Array, low: Array, mask_left: Array) -> Array:
    """
    Function applies the forward transform on dimensions with only a lower bound.

    Parameters:
    -----------
        x: constrained array with shape (N, D).
        low: lower bounds with shape (D,).
        mask_left: mask for left-bounded dimensions.

    Returns:
    --------
        transformed u values for left-bounded dimensions.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    mask_left = jnp.asarray(mask_left, dtype=bool)

    # select only left-bounded dimensions
    x_sel = x[:, mask_left]     # (N, K)
    low_sel = low[mask_left]    # (K,)

    # apply u = log(x - low)
    return jnp.log(x_sel - low_sel)


def _inverse_affine_jax(
    u: Array,                 # (N, D)
    mu: Array,                # (D,)
    sigma: Array,             # (D,)  (use it if diagonal=True)
    L: Array,                 # (D, D) (use it if diagonal=False)
    log_det_L: Array,         # scalar (use it if diagonal=False)
    diagonal: Array | bool,   # scalar bool
) -> tuple[Array, Array]:
    """
    Function applies the inverse affine scaling transform.

    Parameters:
    -----------
        u: input array with shape (N, D).
        mu: mean vector with shape (D,).
        sigma: diagonal scale vector with shape (D,).
        L: Cholesky factor with shape (D, D).
        log_det_L: log-determinant of L.
        diagonal: if True, use diagonal scaling, otherwise use full scaling.

    Returns:
    --------
        transformed x array and log-determinant vector.
    """
    u = jnp.asarray(u)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L = jnp.asarray(L)
    log_det_L = jnp.asarray(log_det_L)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    # build a length-N vector used to broadcast log-determinant
    n = u.shape[0]
    ones_n = jnp.ones((n,), dtype=jnp.result_type(u, mu, sigma, L, log_det_L))

    def _diag_branch(_):
        """
        Function applies diagonal inverse affine scaling.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            transformed x array and log-determinant vector.
        """
        # apply x = mu + sigma * u
        x = mu + sigma * u
        log_det = jnp.sum(jnp.log(sigma)) * ones_n
        return x, log_det

    def _full_branch(_):
        """
        Function applies full inverse affine scaling.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            transformed x array and log-determinant vector.
        """
        # vectorized version of: mu + np.array([L @ ui for ui in u])
        # apply x = mu + u @ L.T.
        x = mu + (u @ L.T)
        log_det = log_det_L * ones_n
        return x, log_det

    # choose diagonal or full affine scaling
    x, log_det = jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)
    return x, log_det


def _forward_affine_jax(
    x: Array,          # (N, D)
    mu: Array,         # (D,)
    sigma: Array,      # (D,)    used if diagonal=True
    L_inv: Array,      # (D, D)  used if diagonal=False
    diagonal: Array,   # scalar bool
) -> Array:
    """
    Function applies the forward affine scaling transform.

    Parameters:
    -----------
        x: input array with shape (N, D).
        mu: mean vector with shape (D,).
        sigma: diagonal scale vector with shape (D,).
        L_inv: inverse Cholesky factor with shape (D, D).
        diagonal: if True, use diagonal scaling, otherwise use full scaling.

    Returns:
    --------
        transformed u array.
    """
    x = jnp.asarray(x)
    mu = jnp.asarray(mu)
    sigma = jnp.asarray(sigma)
    L_inv = jnp.asarray(L_inv)
    diagonal = jnp.asarray(diagonal, dtype=bool)

    def _diag_branch(_):
        """
        Function applies diagonal forward affine scaling.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            transformed u array.
        """
        # apply u = (x - mu) / sigma
        return (x - mu) / sigma

    def _full_branch(_):
        """
        Function applies full forward affine scaling.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            transformed u array.
        """
        # vectorized version of: np.array([L_inv @ (xi - mu) for xi in x])
        # apply u = (x - mu) @ L_inv.T.
        return (x - mu) @ L_inv.T

    # choose diagonal or full affine scaling
    return jax.lax.cond(diagonal, _diag_branch, _full_branch, operand=None)



_LOG_SQRT_2PI = jnp.log(jnp.sqrt(2.0 * jnp.pi))


def _inverse_jax(
    u: jax.Array,             # (N, D)
    low: jax.Array,           # (D,)
    high: jax.Array,          # (D,)
    mask_none: jax.Array,     # (D,) bool
    mask_left: jax.Array,     # (D,) bool
    mask_right: jax.Array,    # (D,) bool
    mask_both: jax.Array,     # (D,) bool
    transform_id: jax.Array,  # scalar int: 0=logit, 1=probit
) -> tuple[jax.Array, jax.Array]:
    """
    Function applies the inverse bounds transform with fixed output shape.

    Parameters:
    -----------
        u: unconstrained array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        mask_none: mask for unbounded dimensions.
        mask_left: mask for left-bounded dimensions.
        mask_right: mask for right-bounded dimensions.
        mask_both: mask for two-sided bounded dimensions.
        transform_id: integer code, 0 for logit and 1 for probit.

    Returns:
    --------
        transformed x array and summed log-determinant vector.
    """
    u = jnp.asarray(u)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # expand masks so they broadcast over rows
    mask_none = jnp.asarray(mask_none, dtype=bool)[None, :]   # (1, D)
    mask_left = jnp.asarray(mask_left, dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both = jnp.asarray(mask_both, dtype=bool)[None, :]

    # initiate transform choice   
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = (transform_id == 1)  # scalar bool 

    # build span and log-span only where both bounds are finite
    span = jnp.where(mask_both[0], high - low, 1.0)           # (D,) 
    log_span = jnp.log(span)                                  # (D,)

    # logit inverse branch for all dimensions
    p_sig = jax.nn.sigmoid(u)                                 # (N, D)
    x_logit = low + p_sig * span                              # (N, D)
    J_logit = log_span + jnp.log(p_sig) + jnp.log1p(-p_sig)    # (N, D)

    # probit inverse branch for all dimensions
    p_phi = jsp.special.ndtr(u)                               # (N, D)
    x_probit = low + p_phi * span                             # (N, D)
    J_probit = log_span + (-0.5 * u * u) - _LOG_SQRT_2PI       # (N, D)

    # select requested two-sided branch
    x_both = jnp.where(is_probit, x_probit, x_logit)           # (N, D)
    J_both = jnp.where(is_probit, J_probit, J_logit)           # (N, D)

    # one-sided branches (computed for all dims, only used where their mask=True)
    exp_u = jnp.exp(u)
    x_left = exp_u + low
    J_left = u

    x_right = high - exp_u
    J_right = u

    # create full (N, D) output arrays with static shapes
    x = jnp.zeros_like(u)
    J = jnp.zeros_like(u)

    x = jnp.where(mask_none, u, x)
    x = jnp.where(mask_left, x_left, x)
    x = jnp.where(mask_right, x_right, x)
    x = jnp.where(mask_both, x_both, x)

    # mask_none contributes 0 to J, so we only set left/right/both
    J = jnp.where(mask_left, J_left, J)
    J = jnp.where(mask_right, J_right, J)
    J = jnp.where(mask_both, J_both, J)

    # sum diagonal log-Jacobian terms over dimensions
    log_det_J = jnp.sum(J, axis=1)  # (N,)
    return x, log_det_J


def _forward_jax(
    x: Array,                  # (N, D)
    low: Array,                # (D,)
    high: Array,               # (D,)
    mask_none: Array,          # (D,) bool
    mask_left: Array,          # (D,) bool
    mask_right: Array,         # (D,) bool
    mask_both: Array,          # (D,) bool
    transform_id: Array,       # scalar int: 0=logit, 1=probit
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Function applies the forward bounds transform with fixed output shape.

    Parameters:
    -----------
        x: constrained array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        mask_none: mask for unbounded dimensions.
        mask_left: mask for left-bounded dimensions.
        mask_right: mask for right-bounded dimensions.
        mask_both: mask for two-sided bounded dimensions.
        transform_id: integer code, 0 for logit and 1 for probit.
        eps: clipping value used for numerical stability.

    Returns:
    --------
        transformed u array with shape (N, D).
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)

    # expand masks so they broadcast over rows
    mask_none  = jnp.asarray(mask_none,  dtype=bool)[None, :]
    mask_left  = jnp.asarray(mask_left,  dtype=bool)[None, :]
    mask_right = jnp.asarray(mask_right, dtype=bool)[None, :]
    mask_both  = jnp.asarray(mask_both,  dtype=bool)[None, :]

    # read transform choice
    transform_id = jnp.asarray(transform_id, dtype=jnp.int64)
    is_probit = (transform_id == 1)

    # unbounded dimensions pass through unchanged
    u_none = x

    # left: log(x - low)
    u_left = jnp.log(x - low)

    # one-sided forward transforms: right: log(high - x)
    u_right = jnp.log(high - x)

    # two-sided probabilities: p=(x-low)/(high-low) then logit/probit
    span = jnp.where(mask_both[0], high - low, 1.0)       # (D,)
    low_safe = jnp.where(mask_both[0], low, 0.0)          # (D,)
    p = (x - low_safe) / span                              # (N, D)

    # clip probabilities away from 0 and 1
    eps_t = jnp.asarray(eps, dtype=x.dtype)
    p = jnp.clip(p, eps_t, 1.0 - eps_t)

    # initialize two-sided logit and probit transforms
    u_logit = jnp.log(p) - jnp.log1p(-p)
    u_probit = jnp.sqrt(jnp.asarray(2.0, dtype=x.dtype)) * jsp.special.erfinv(2.0 * p - 1.0)
    u_both = jnp.where(is_probit, u_probit, u_logit)

    # assemble full output array (N, D)
    u = jnp.zeros_like(x)
    u = jnp.where(mask_none,  u_none,  u)
    u = jnp.where(mask_left,  u_left,  u)
    u = jnp.where(mask_right, u_right, u)
    u = jnp.where(mask_both,  u_both,  u)
    return u


def inverse_jax(u: Array, cfg: Mapping[str, Array], masks: Mapping[str, Array]) -> tuple[Array, Array]:
    """
    Function applies the full inverse transform, including optional affine scaling.

    Parameters:
    -----------
        u: unconstrained input with shape (N, D).
        cfg: configuration dictionary with bounds and scaling values.
             Must contain: low, high, transform_id, scale, diagonal, 
             mu, sigma, L, log_det_L
        masks: dictionary with bound masks. Must contain: mask_none, 
             mask_left, mask_right, mask_both

    Returns:
    --------
        transformed:
            x : Array, shape (N, D)
            log-determinant vector: log_det_J : Array, shape (N,)
    """
    u = jnp.asarray(u)

    # values from the configuration and mask dictionaries
    low = cfg["low"]
    high = cfg["high"]
    transform_id = cfg["transform_id"]

    scale = jnp.asarray(cfg["scale"], dtype=bool)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    mu = cfg["mu"]
    sigma = cfg["sigma"]
    L = cfg["L"]
    log_det_L = cfg["log_det_L"]

    mask_none = masks["mask_none"]
    mask_left = masks["mask_left"]
    mask_right = masks["mask_right"]
    mask_both = masks["mask_both"]

    def _scaled(u_in: Array) -> tuple[Array, Array]:
        """
        Function applies inverse affine scaling and then inverse bounds transform.

        Parameters:
        -----------
            u_in: unconstrained input array.

        Returns:
        --------
            transformed x array and total log-determinant.
        """
        # undo affine scaling, then undo bounds transform
        x1, ld1 = _inverse_affine_jax(u_in, mu, sigma, L, log_det_L, diagonal)
        x2, ld2 = _inverse_jax(
            x1, low, high,
            mask_none, mask_left, mask_right, mask_both,
            transform_id,
        )
        return x2, ld1 + ld2

    def _unscaled(u_in: Array) -> tuple[Array, Array]:
        """
        Function applies only the inverse bounds transform.

        Parameters:
        -----------
            u_in: unconstrained input array.

        Returns:
        --------
            transformed x array and log-determinant.
        """
        # skip affine scaling when it is disabled
        return _inverse_jax(
            u_in, low, high,
            mask_none, mask_left, mask_right, mask_both,
            transform_id,
        )

    # choose scaled or unscaled inverse transformation
    x, log_det_J = jax.lax.cond(scale, _scaled, _unscaled, u)
    return x, log_det_J


def forward_jax(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Function applies the full forward transform, including optional affine scaling.

    Parameters:
    -----------
        x: constrained input with shape (N, D).
        cfg: configuration dictionary with bounds and scaling values.
        masks: dictionary with bound masks.
        eps: clipping value used for numerical stability.

    Returns:
    --------
        transformed u array.
    """
    x = jnp.asarray(x)

    # apply the bounds transform first
    u0 = _forward_jax(
        x,
        cfg["low"], cfg["high"],
        masks["mask_none"], masks["mask_left"], masks["mask_right"], masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    # affine-scaling options
    scale = jnp.asarray(cfg["scale"], dtype=bool)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    def _scaled(u_in: Array) -> Array:
        """
        Function applies the forward affine scaling.

        Parameters:
        -----------
            u_in: bounds-transformed array.

        Returns:
        --------
            fully transformed u array.
        """
        # apply affine scaling when it is enabled
        return _forward_affine_jax(u_in, cfg["mu"], cfg["sigma"], cfg["L_inv"], diagonal)

    # choose scaled or unscaled forward transformation
    return jax.lax.cond(scale, _scaled, lambda z: z, u0)



def forward_jax_checked(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
) -> Array:
    """
    Function checks bounds and then applies the full forward transform.

    Parameters:
    -----------
        x: constrained input with shape (N, D).
        cfg: configuration dictionary with bounds and scaling values.
        masks: dictionary with bound masks.
        eps: clipping value used for numerical stability.

    Returns:
    --------
        transformed u array.
    """
    # check that x stays inside the configured bounds
    x = assert_array_within_interval(x, cfg["low"], cfg["high"], name="x")
    return forward_jax(x, cfg, masks, eps=eps)


def fit_jax(
    x: Array,
    cfg: Mapping[str, Array],
    masks: Mapping[str, Array],
    *,
    eps: float = 1e-13,
    jitter: float = 0.0,   # set 1e-6 if problems with Cholevsky
) -> dict[str, Array]:
    """
    Function fits the affine scaling parameters after the bounds transform.

    Parameters:
    -----------
        x: constrained input with shape (N, D).
        cfg: configuration dictionary with bounds and scaling placeholders.
        masks: dictionary with bound masks.
        eps: clipping value used for the bounds transform.
        jitter: diagonal jitter added before Cholesky in the full-covariance case.

    Returns:
    --------
        new configuration dictionary with fitted affine parameters.
    """
    x = jnp.asarray(x)

    # (i) forward bounds transform before fitting affine scaling
    u = _forward_jax(
        x,
        cfg["low"], cfg["high"],
        masks["mask_none"], masks["mask_left"], masks["mask_right"], masks["mask_both"],
        cfg["transform_id"],
        eps=eps,
    )

    # compute mean of the transformed data
    mu = jnp.mean(u, axis=0)
    diagonal = jnp.asarray(cfg["diagonal"], dtype=bool)

    # build common constants for diagonal and full branches
    D = u.shape[1]
    dtype = u.dtype
    I = jnp.eye(D, dtype=dtype)
    zero = jnp.asarray(0.0, dtype=dtype)

    def _diag_branch(_):
        """
        Function fits diagonal affine scaling.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            sigma, cov, L, L_inv, and log_det_L values for the diagonal case.
        """
        # use per-dimension standard deviation
        sigma = jnp.std(u, axis=0)   # ddof=0 (matches np.std default)
        cov = I
        L = I
        L_inv = I
        log_det_L = zero
        return sigma, cov, L, L_inv, log_det_L

    def _full_branch(_):
        """
        Function fits full affine scaling from the covariance matrix.

        Parameters:
        -----------
            _: unused operand.

        Returns:
        --------
            sigma, cov, L, L_inv, and log_det_L values for the full case.
        """
        # compute sample covariance of transformed data:
        # np.cov(u.T) equivalent: centered.T @ centered / (N-1)
        n = u.shape[0]
        denom = jnp.asarray(jnp.maximum(n - 1, 1), dtype=dtype)  # avoid divide-by-zero if n==1
        centered = u - mu
        cov = (centered.T @ centered) / denom

        # numerical stabilization: optional jitter before Cholesky
        cov = cov + jnp.asarray(jitter, dtype=dtype) * I

        # factorize covariance and build its inverse factor
        L = jnp.linalg.cholesky(cov)

        # L_inv = inv(L)
        L_inv = jsp.linalg.solve_triangular(L, I, lower=True)

        # log(det(L)) for triangular L
        log_det_L = jnp.sum(jnp.log(jnp.diag(L)))

        # keep sigma for consistency with config structure
        sigma = cfg.get("sigma", jnp.ones((D,), dtype=dtype))
        return sigma, cov, L, L_inv, log_det_L

    # choose diagonal or full affine fitting
    sigma, cov, L, L_inv, log_det_L = jax.lax.cond(
        diagonal, _diag_branch, _full_branch, operand=None
    )

    # return new config dictionary with updated fitted values
    cfg_out = dict(cfg)
    cfg_out.update(
        mu=mu,
        sigma=sigma,
        cov=cov,
        L=L,
        L_inv=L_inv,
        log_det_L=log_det_L,
    )
    return cfg_out



def apply_reflective_boundary_conditions_x_jax(
    x: Array,
    low: Array,
    high: Array,
    reflective_mask: Array,
) -> Array:
    """
    Function applies reflective boundary conditions to selected dimensions.

    For each reflective dimension i with finite bounds [low[i], high[i]],
    values are reflected back into the interval, equivalent to repeatedly applying:
      while x > high: x = 2*high - x
      while x < low:  x = 2*low  - x

    Parameters:
    -----------
        x: input array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        reflective_mask: mask for reflective dimensions.

    Returns:
    --------
    array with reflected values on reflective dimensions.
        x_ref : Array, shape (N, D)
        Reflected x (unchanged on non-reflective dims).
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    reflective_mask = jnp.asarray(reflective_mask, dtype=bool)

    # skip when no reflective dimensions exist
    has_reflect = jnp.any(reflective_mask)

    def _do_reflect(x_in: Array) -> Array:
        """
        Function reflects values back into their intervals.

        Parameters:
        -----------
            x_in: input array with shape (N, D).

        Returns:
        --------
            reflected array.
        """
        # expand mask across rows
        m = reflective_mask[None, :]  # (1, D)

        # dont touch non-reflective dims while computing (prevents inf/nan propagation)
        x_safe = jnp.where(m, x_in, 0.0)

        # build safe lower bounds and interval widths
        low_safe = jnp.where(reflective_mask, low, 0.0)     # (D,)
        span = jnp.where(reflective_mask, high - low, 1.0)  # (D,)

        # keep bound numerically stable: guard against zero-width intervals
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        # define period
        period = 2.0 * span  # (D,)

        # fold into a period: [0, 2*span)
        y = jnp.mod(x_safe - low_safe, period)  # (N, D), in [0, period)

        # reflect second half back into interval: [span, 2*span) into [span, 0]
        y = jnp.where(y > span, period - y, y)

        # restore non-reflective dimensions unchanged
        x_ref = low_safe + y  # (N, D)
        return jnp.where(m, x_ref, x_in)

    # apply reflection only when needed
    return jax.lax.cond(has_reflect, _do_reflect, lambda z: z, x)


def apply_periodic_boundary_conditions_x_jax(
    x: Array,
    low: Array,
    high: Array,
    periodic_mask: Array,
) -> Array:
    """
    Function applies periodic boundary conditions to selected dimensions.
        while x > high: x = low + x - high   (subtract period)
        while x < low:  x = high + x - low   (add period)

    Parameters:
    -----------
        x: input array with shape (N, D).
        low: lower bounds with shape (D,).
        high: upper bounds with shape (D,).
        periodic_mask: mask for periodic dimensions.

    Returns:
    --------
        array with wrapped values on periodic dimensions.
    """
    x = jnp.asarray(x)
    low = jnp.asarray(low)
    high = jnp.asarray(high)
    periodic_mask = jnp.asarray(periodic_mask, dtype=bool)

    # skip when no periodic dimensions exist
    has_periodic = jnp.any(periodic_mask)

    def _wrap(x_in: Array) -> Array:
        """
        Function wraps values back into their periodic intervals.

        Parameters:
        -----------
            x_in: input array with shape (N, D).

        Returns:
        --------
            wrapped array.
        """
        # expand mask across rows and keep only finite intervals
        m = periodic_mask[None, :]  # (1, D)

        # periodic for finite bounds only
        fin = jnp.isfinite(low) & jnp.isfinite(high)
        m = m & fin[None, :]

        # placeholders to avoid inf/nan propagation in non-periodic dims
        x_safe = jnp.where(m, x_in, 0.0)

        # safe lower bounds, upper bounds, and interval widths
        low_safe = jnp.where(periodic_mask, low, 0.0)     # (D,)
        high_safe = jnp.where(periodic_mask, high, 1.0)   # (D,)
        span = jnp.where(periodic_mask, high - low, 1.0)  # (D,)

        # protect against zero-width intervals
        tiny = jnp.asarray(jnp.finfo(x_in.dtype).tiny, dtype=x_in.dtype)
        span = jnp.where(span > tiny, span, 1.0)

        # wrap to [low, high)
        y = jnp.mod(x_safe - low_safe, span)              # in [0, span)
        x_wrap = low_safe + y                             # in [low, high)

        # map positive multiples to 'high' instead of 'low'
        pos = (x_safe - low_safe) > 0                     # excludes x==low, includes x==high and above
        x_wrap = jnp.where((y == 0.0) & pos, high_safe, x_wrap)

        # restore non-periodic dimensions unchanged
        return jnp.where(m, x_wrap, x_in)

    # apply wrapping only when needed
    return jax.lax.cond(has_periodic, _wrap, lambda z: z, x)



def apply_boundary_conditions_x_jax(
    x: Array,
    cfg: dict[str, Array],
) -> Array:
    """
    Function applies periodic and reflective boundary conditions to x.

    Parameters:
    -----------
        x: input array with shape (N, D).
        cfg: configuration dictionary with bounds and boundary masks.

    Returns:
    --------
        array after periodic and reflective boundary handling.
    """
    x = jnp.asarray(x)

    # bounds and boundary masks from config
    low = cfg["low"]
    high = cfg["high"]
    periodic_mask = jnp.asarray(cfg["periodic_mask"], dtype=bool)
    reflective_mask = jnp.asarray(cfg["reflective_mask"], dtype=bool)

    # check whether each type of boundary rule is present
    has_periodic = jnp.any(periodic_mask)
    has_reflective = jnp.any(reflective_mask)

    def _apply_periodic(x_in: Array) -> Array:
        """
        Function applies periodic boundary handling.

        Parameters:
        -----------
            x_in: input array.

        Returns:
        --------
            array after periodic wrapping.
        """
        # apply periodic boundary rule
        return apply_periodic_boundary_conditions_x_jax(x_in, low, high, periodic_mask)

    def _apply_reflective(x_in: Array) -> Array:
        """
        Function applies reflective boundary handling.

        Parameters:
        -----------
            x_in: input array.

        Returns:
        --------
            array after reflective wrapping.
        """
        # apply reflective boundary rule
        return apply_reflective_boundary_conditions_x_jax(x_in, low, high, reflective_mask)

    # apply periodic first
    x1 = jax.lax.cond(has_periodic, _apply_periodic, lambda z: z, x)
    # then apply reflective second
    x2 = jax.lax.cond(has_reflective, _apply_reflective, lambda z: z, x1)

    return x2









