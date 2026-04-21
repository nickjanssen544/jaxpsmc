from functools import partial
import jax.numpy as jnp
from jax import lax
import jax
jax.config.update("jax_enable_x64", True)
print("x64 enabled?", jax.config.jax_enable_x64)



@partial(jax.jit, static_argnames=("bins",))
def trim_weights_jax(samples, weights, ess=0.99, bins=1000):
    """
    Function trims importance weights by scanning percentile thresholds.

    Parameters:
    -----------
        samples: 
            sample index array used only to match the original interface.
        weights: 
            input weight array.
        ess: 
            target ESS ratio after trimming.
        bins: 
            number of percentile grid points.

    Returns:
    --------
        trim mask: 
            (N,) bool
        weights_trimmed: 
            (N,) weights after trimming, renormalized (zeros for dropped)
        threshold: 
            scalar weight threshold is used here
        ess_ratio:       
            ess_trimmed / ess_total
        i_final: 
            index in percentile grid that was selected
        
        
        trimmed weights, threshold, ESS ratio, and selected grid index.
    """
    samples = jnp.asarray(samples)
    weights = jnp.asarray(weights)

    dtype = jnp.result_type(weights, jnp.asarray(ess))
    weights = weights.astype(dtype)
    ess = jnp.asarray(ess, dtype=dtype)

    # normalize weights
    wsum = jnp.sum(weights)
    bad = (wsum <= 0) | jnp.isnan(wsum)

    w = weights / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # compute ESS before trimming
    ess_total = 1.0 / jnp.sum(w * w)

    # define percentile grid and sorted weights
    percentiles = jnp.linspace(jnp.asarray(0.0, dtype), jnp.asarray(99.0, dtype), bins)
    sorted_w = jnp.sort(w)  # 

    # nr of weights
    n = w.shape[0]
    n_minus_1 = jnp.asarray(n - 1, dtype)

    def stats_for_i(i):
        """
        Function computes trimming statistics for one percentile index.

        Parameters:
        -----------
            i: percentile grid index.

        Returns:
        --------
            threshold, trim mask, trimmed weights, and ESS ratio.
        """
        # current percentile: p in [0, 99]
        p = lax.dynamic_index_in_dim(percentiles, i, axis=0, keepdims=False)
        frac = p / jnp.asarray(100.0, dtype)

        # linear interpolation percentile threshold from sorted weights
        pos = frac * n_minus_1                    # in [0, n-1]
        lo = jnp.floor(pos).astype(jnp.int32)
        hi = jnp.minimum(lo + 1, jnp.int32(n - 1))
        alpha = pos - lo.astype(dtype)

        w_lo = sorted_w[lo]
        w_hi = sorted_w[hi]
        threshold = (1.0 - alpha) * w_lo + alpha * w_hi

        # keep only weights above threshold
        mask = w >= threshold
        w_kept = jnp.where(mask, w, 0.0)
        kept_sum = jnp.sum(w_kept)

        # renormalize kept weights
        kept_sum_safe = jnp.where(kept_sum > 0, kept_sum, jnp.asarray(1.0, dtype))
        w_trim = jnp.where(mask, w_kept / kept_sum_safe, 0.0)

        # compute ESS ratio after trimming     
        ess_trim = 1.0 / jnp.sum(w_trim * w_trim)
        ratio = ess_trim / ess_total
        return threshold, mask, w_trim, ratio

    # search from high percentile downward until ratio >= ess (or is zero)
    def cond(state):
        """
        Function checks whether the threshold search should continue.

        Parameters:
        -----------
            state: current percentile index and done flag.

        Returns:
        --------
            boolean value that is True while the search should continue.
        """
        # continue until valid threshold has been found
        i, done = state
        return ~done

    def body(state):
        """
        Function performs one threshold-search step.

        Parameters:
        -----------
            state: current percentile index and done flag.

        Returns:
        --------
            updated percentile index and done flag.
        """
        # current search state
        i, done = state

        # # check if current percentile reaches target ESS ratio
        _, _, _, ratio = stats_for_i(i)
        satisfied = ratio >= ess

        # stop when target is reached or grid start is reached
        done2 = done | satisfied | (i == 0)
        i2 = jnp.where(done2, i, i - 1)
        return i2, done2

    # start from largest percentile and move downward
    i0 = jnp.int32(bins - 1)
    i_final, _ = lax.while_loop(cond, body, (i0, False))

    # rebuild trimming result for  selected index
    threshold, mask, w_trim, ratio = stats_for_i(i_final)

    # if weights were invalid, return "empty outputs" + NaNs for weights_trimmed
    mask = jnp.where(bad, jnp.zeros_like(mask), mask)
    w_trim = jnp.where(bad, jnp.full_like(w_trim, jnp.nan), w_trim)
    threshold = jnp.where(bad, jnp.asarray(jnp.nan, dtype), threshold)
    ratio = jnp.where(bad, jnp.asarray(jnp.nan, dtype), ratio)

    return mask, w_trim, threshold, ratio, i_final


@jax.jit
def effective_sample_size_jax(weights):
    """
    Function computes the effective sample size from a weight vector.

    Parameters:
    -----------
        weights: input weight array.

    Returns:
    --------
        effective sample size, or NaN if the input is invalid.
    """
    w = jnp.asarray(weights)
    wsum = jnp.sum(w)

    # invalid inputs if sum<=0 or non-finite
    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)
    ess = 1.0 / jnp.sum(w_norm * w_norm)

    # return NaN for invalid inputs
    return jnp.where(bad, jnp.asarray(jnp.nan, ess.dtype), ess)


@jax.jit
def unique_sample_size_jax(weights, k=-1):
    """
    Function computes the unique sample size from a weight vector.

    Parameters:
    -----------
        weights: input weight array with shape (N,) or (..., N).
        k: number of draws used in the formula. If k < 0, the function uses N.

    Returns:
    --------
        unique sample size: 
            scalar if weights is (N,), or array (...) if weights is (..., N).
            or
            NaN where weights are invalid (sum<=0 or non-finite).

    """ 
    w = jnp.asarray(weights)
    wsum = jnp.sum(w, axis=-1, keepdims=True)

    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights without mutation
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, w.dtype), wsum)

    # choose k from input or from last axis length
    N = w.shape[-1]
    k_eff = lax.cond(k < 0, lambda _: jnp.int32(N), lambda _: jnp.int32(k), operand=None)

    # compute unique sample size formula: sum_i [ 1 - (1 - w_i)^k ]
    # works for k=0 too then term becomes 0
    term = 1.0 - jnp.power(1.0 - w_norm, k_eff)
    uss = jnp.sum(term, axis=-1)

    # return NaN for invalid inputs
    uss = jnp.where(jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, uss.dtype), uss)
    return uss


@jax.jit
def compute_ess_jax(logw):
    """
    Function computes the ESS fraction from log-weights.
    Compute ESS fraction = (1 / sum(w^2)) / N, with w = softmax(logw).

    Parameters:
    -----------
        logw: input log-weight array with shape (N,) or (..., N).

    Returns:
    --------
        ESS:
            ess_frac: scalar if logw is (N,)
            or
            array (...) if logw is (..., N).
            or
            NaN if inputs are non-finite
    """
    lw = jnp.asarray(logw)

    # stabilize exponentials by subtracting maximum
    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # convert log-weights into normalized weights:
    # exponentiate and normalize using softmax weights
    w_unnorm = jnp.exp(lw0)
    wsum = jnp.sum(w_unnorm, axis=-1, keepdims=True)

    # invalid inputs
    bad = (wsum <= 0) | jnp.isnan(wsum) | jnp.isinf(wsum)

    # normalize weights and compute ESS
    w = w_unnorm / jnp.where(bad, jnp.asarray(1.0, w_unnorm.dtype), wsum)
    ess = 1.0 / jnp.sum(w * w, axis=-1)          # ESS

    # divide ESS by nr of weights
    N = lw.shape[-1]
    ess_frac = ess / jnp.asarray(N, ess.dtype)   # ESS / N

    # NaN for invalid inputs
    ess_frac = jnp.where(jnp.squeeze(bad, axis=-1), jnp.asarray(jnp.nan, ess_frac.dtype), ess_frac)
    return ess_frac


@jax.jit
def increment_logz_jax(logw):
    """
    Function computes the log-sum-exp of a log-weight array.
    logZ increment: logsumexp(logw)

    Parameters:
    -----------
        logw: input log-weight array with shape (N,) or (..., N).

    Returns:
    --------
        log-sum-exp value, or NaN if the result is not finite.
            logz_inc: scalar if logw is (N,), or array (...) if logw is (..., N).
            NaN if inputs are all -inf
    """
    lw = jnp.asarray(logw)

    lw_max = jnp.max(lw, axis=-1, keepdims=True)
    lw0 = lw - lw_max

    # compute logsumexp as max plus log of the summed exponentials: 
    # logsumexp = max + log(sum(exp(lw - max)))
    lse = lw_max + jnp.log(jnp.sum(jnp.exp(lw0), axis=-1, keepdims=True))

    # remove last singleton axis
    lse = jnp.squeeze(lse, axis=-1)
    # NaN when the result is not finite
    lse = jnp.where(jnp.isfinite(lse), lse, jnp.nan)

    return lse


_ECONVERGED = jnp.int64(0)
_EVALUEERR  = jnp.int64(-3)

@partial(jax.jit, static_argnames=("size",))
def _systematic_resample_impl(key, weights, size: int):
    """
    Function runs the core systematic resampling algorithm.

    Parameters:
    -----------
        key: JAX random key.
        weights: input weight array.
        size: number of indices to draw.

    Returns:
    --------
        resampled indices, status code, and output key.
    """
    w = jnp.asarray(weights)
    dtype = jnp.result_type(w, jnp.float64)

    # weights validation
    wsum = jnp.sum(w)
    bad = (wsum <= 0) | (~jnp.isfinite(wsum)) | jnp.any(~jnp.isfinite(w)) | jnp.any(w < 0)
    # weights normalization
    w_norm = w / jnp.where(bad, jnp.asarray(1.0, dtype), wsum)

    # cumulative distribution function
    cdf = jnp.cumsum(w_norm)
    cdf = cdf / jnp.where(bad, jnp.asarray(1.0, dtype), cdf[-1])

    # draw one random offset and define systematic positions
    key_out, subkey = jax.random.split(key)
    u = jax.random.uniform(subkey, shape=(), dtype=dtype)
    positions = (u + jnp.arange(size, dtype=dtype)) / jnp.asarray(size, dtype=dtype)

    # dummy indices when you have invalid weights
    idx = jnp.searchsorted(cdf, positions, side="left")
    idx = jnp.clip(idx, 0, w.shape[0] - 1).astype(jnp.int32)

    idx = jnp.where(bad, jnp.full((size,), jnp.int32(-1)), idx)
    status = jnp.where(bad, _EVALUEERR, _ECONVERGED)
    return idx, status, key_out


def systematic_resample_jax(weights, *, key):
    """
    Function resamples exactly len(weights) indices by systematic resampling.

    Parameters:
    -----------
        weights: input weight array.
        key: JAX random key.

    Returns:
    --------
        resampled indices, status code, and output key.
    """
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, w.shape[0])


def systematic_resample_jax_size(weights, *, key, size: int):
    """
    Function resamples a fixed number of indices by systematic resampling.

    Parameters:
    -----------
        weights: input weight array.
        key: JAX random key.
        size: number of indices to draw.

    Returns:
    --------
        resampled indices, status code, and output key.
    """
    w = jnp.asarray(weights)
    return _systematic_resample_impl(key, w, size)



