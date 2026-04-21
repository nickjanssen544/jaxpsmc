from functools import partial
import jax
import jax.numpy as jnp
from jax import lax
jax.config.update("jax_enable_x64", True)

_ECONVERGED  = jnp.int32(0)
_ESIGNERR    = jnp.int32(-1)
_ECONVERR    = jnp.int32(-2)
_EVALUEERR   = jnp.int32(-3)

def _bisect_impl(f, a, b, *, xtol, rtol, maxiter, args):
    """
    Function runs the core bisection method for one interval.

    Parameters:
    -----------
    f: 
        function to solve. It should return a scalar value.
    a: 
        left endpoint of the interval.
    b: 
        right endpoint of the interval.
    xtol: 
        absolute tolerance for interval width.
    rtol: 
        relative tolerance based on current midpoint.
    maxiter: 
        maximum number of loop iterations.
    args: 
        any extra arguments to f.

    Returns:
    -------
    x: 
        estimated root if valid input
        NaN if invalid input.
    status: 
        integer describing success or failure.
    it: 
        number of iterations used.
    funcalls: 
        number of function calls used.
    """
    # convert inputs to JAX arrays
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    # pick one dtype for all values
    dtype = jnp.result_type(a, b, jnp.asarray(xtol), jnp.asarray(rtol))
    a = a.astype(dtype)
    b = b.astype(dtype)
    xtol = jnp.asarray(xtol, dtype=dtype)
    rtol = jnp.asarray(rtol, dtype=dtype)
    maxiter = jnp.asarray(maxiter, dtype=jnp.int32)

    # maxiter is positive
    bad_maxiter = maxiter < 0

    # evaluate the function at both endpoints
    fa = f(a, *args)
    fb = f(b, *args)
    funcalls0 = jnp.int32(2)
    it0 = jnp.int32(0)

    # check if endpoint is already root
    a_is_root = (fa == jnp.asarray(0, dtype=dtype))
    b_is_root = (fb == jnp.asarray(0, dtype=dtype))
    converged0 = a_is_root | b_is_root

    # stop early if function values are NaN
    any_nan0 = jnp.isnan(fa) | jnp.isnan(fb)
    # bisection needs opposite signs at endpoints
    bracketed0 = (jnp.sign(fa) != jnp.sign(fb))

    # start from midpoint unless one endpoint is root
    x0 = jnp.asarray(0.5, dtype=dtype) * (a + b)
    x0 = jnp.where(a_is_root, a, jnp.where(b_is_root, b, x0))

    # only do loop if everything above is valid
    need_loop0 = (~bad_maxiter) & (~any_nan0) & (~converged0) & bracketed0 & (maxiter > 0)
    nan_seen0 = any_nan0

    # keep active interval and endpoint values
    left, right = a, b
    fleft, fright = fa, fb


    def cond(state):
        """
        Function decides whether the bisection loop should continue.

        Parameters:
        -----------
        state: 
            tuple with current interval, function values, counters,
            and status flags.

        Returns:
        -------
        Boolean value which tells whether to run another loop step.
        """
        # unpack the current loop state
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state
        # continue only if midpoint is not a root
        return need_loop & (~converged) & (~nan_seen) & (it < maxiter)


    def body(state):
        """
        Function performs one bisection step.

        Parameters:
        -----------
        state: 
            tuple with current interval, function values, counters,
            and status flags.

        Returns:
        -------
        Updated state after one midpoint evaluation and interval update.
        """
        # unpack the current loop state.
        left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = state

        # count iteration
        it = it + jnp.int32(1)

        # compute midpoint and evaluate the function         
        x = jnp.asarray(0.5, dtype=dtype) * (left + right)
        fx = f(x, *args)
        funcalls = funcalls + jnp.int32(1)

        # track any invalid values
        fx_nan = jnp.isnan(fx)
        nan_seen = nan_seen | fx_nan

        # check if midpoint is exactly a root       
        fx_is_root = (fx == jnp.asarray(0, dtype=dtype))

        # decide which half of the interval still contains root
        same_sign = (jnp.sign(fleft) == jnp.sign(fx))
        go_left = same_sign & (~fx_is_root) & (~fx_nan)

        # update interval
        left2  = jnp.where(go_left, x, left)
        fleft2 = jnp.where(go_left, fx, fleft)
        right2  = jnp.where(go_left, right, x)
        fright2 = jnp.where(go_left, fright, fx)

        # check stopping condition based on interval width
        width = jnp.abs(right2 - left2)
        tol = xtol + rtol * jnp.abs(x)
        converged = fx_is_root | (width <= tol)

        return (left2, right2, fleft2, fright2, x, it, funcalls, converged, nan_seen, need_loop)

    # run bisection loop in JAX
    left, right, fleft, fright, x, it, funcalls, converged, nan_seen, need_loop = lax.while_loop(
        cond,
        body,
        (left, right, fleft, fright, x0, it0, funcalls0, converged0, nan_seen0, need_loop0),
    )

    # build status code
    status = lax.select(
        bad_maxiter,
        _EVALUEERR,
        lax.select(
            nan_seen,
            _EVALUEERR,
            lax.select(
                converged0 | (need_loop0 & converged),
                _ECONVERGED,
                lax.select((~any_nan0) & (~converged0) & (~bracketed0), _ESIGNERR, _ECONVERR),
            ),
        ),
    )

    # If failure, return NaN instead of a root estimate.
    x = jnp.where(status == _ECONVERGED, x, jnp.nan)
    return x, status, it, funcalls

# jit solver (single interval)
_bisect_jit = jax.jit(_bisect_impl, static_argnames=("f",))




def bisect_jax(
    f, a, b, *,
    xtol=2e-12,
    rtol=4 * jnp.finfo(jnp.float64).eps,
    maxiter=100,
    args=(),
):
    """
    Function solves for one root on one interval using bisection.

    Parameters:
    -----------
    f: 
        function to solve, returns a scalar value.
    a:
        left endpoint of the interval.
    b:
        right endpoint of the interval.
    xtol:
        absolute tolerance for stopping.
    rtol: 
        relative tolerance for stopping.
    maxiter: 
        maximum number of iterations.
    args: 
        extra arguments passed to f.

    Returns:
    -------
    tuple: 
        (root, status, iterations, function_calls).
    """
    return _bisect_jit(f, a, b, xtol=xtol, rtol=rtol, maxiter=maxiter, args=args)




def bisect_jax_batch(f, a, b, *, args=(), **kwargs):
    """
    Function runs bisection over many intervals at once with vmap.

    Parameters:
    -----------
    f: 
        function to solve.
    a: 
        array of left endpoints.
    b: 
        array of right endpoints.
    args: 
        extra arguments.
    **kwargs: 
        extra keyword arguments forwarded to bisect_jax.

    Returns:
    Batched tuple of results from bisect_jax for each interval.
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)

    
    def _axis_for(arg):
        arg = jnp.asarray(arg)
        return 0 if arg.ndim > 0 else None

    args_axes = tuple(_axis_for(arg) for arg in args)

    def solve_one(ai, bi, *args_i):
        return bisect_jax(f, ai, bi, args=args_i, **kwargs)

    in_axes = (0, 0) + args_axes
    return jax.vmap(solve_one, in_axes=in_axes)(a, b, *args)












