from __future__ import annotations

from typing import Callable, Mapping, Tuple, Any, Optional, Dict

import jax
import jax.numpy as jnp

from .scaler_jax import *


Array = jax.Array

# helpers assume the flow is used to move between u-space and theta-space.
def _flow_u_to_theta_jax(flow, u: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
    """
    Function maps u values into theta values using flow

    Parameters:
    -----------
    flow: 
        flow object with a bijection.
    u: 
        input array in u-space.
    condition: 
        optional conditioning array passed to the flow.

    Returns:
    --------
    theta: 
        transformed array in theta-space.
    logdet: 
        log absolute determinant of du / dtheta.
    """
    theta, fwd_logdet = flow.bijection.transform_and_log_det(u, condition)
    return theta, -fwd_logdet


def _flow_theta_to_u_jax(flow, theta: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
    """
    Function maps theta values into u values using the flow.

    Parameters:
    -----------
    flow: 
        flow object with a bijection.
    theta: 
        input array in theta-space.
    condition: 
        optional conditioning array passed to the flow.

    Returns:
    --------
    u: 
        transformed array in u-space.
    logdet: 
        log absolute determinant of du / dtheta.
    """
    u, inv_logdet = flow.bijection.inverse_and_log_det(theta, condition)
    return u, inv_logdet


def preconditioned_pcn_jax(
    key: Array,
    *,
    # current state (all arrays; no None)
    u: Array,                 # (N, D)
    x: Array,                 # (N, D)
    logdetj: Array,           # (N,)
    logl: Array,              # (N,)
    logp: Array,              # (N,)
    logdetj_flow: Array,      # (N,)
    blobs: Array,             # (N, B...) ; use shape (N, 0) if no blobs
    beta: Array,              # scalar
    # functions
    loglike_fn: Callable[[Array], Tuple[Array, Array]],
    logprior_fn: Callable[[Array], Array],
    flow: Any,                # FlowJAX Transformed-like object with .bijection
    scaler_cfg: Mapping[str, Array],
    scaler_masks: Mapping[str, Array],
    # geometry (Student-t)
    geom_mu: Array,           # (D,)
    geom_cov: Array,          # (D, D)
    geom_nu: Array,           # scalar
    # options
    n_max: int,
    n_steps: int,
    proposal_scale: Array,    # scalar
    condition: Optional[Array] = None,
) -> Dict[str, Array]:
    """
    This function runs the preconditioned PCN update in JAX

    Parameters:
    ------------
        key: 
            JAX random key.
        u: 
            current particle values in u-space, shape (N, D).
        x: 
            current particle values in x-space, shape (N, D).
        logdetj: 
            current scaler log determinant values, shape (N,).
        logl: 
            current log-likelihood values, shape (N,).
        logp: 
            current log-prior values, shape (N,).
        logdetj_flow: 
            current flow log determinant values, shape (N,).
        blobs: 
            current extra outputs, shape (N, B...) or (N, 0).
        beta: 
            tempering value.
        loglike_fn: 
            function that maps one x value to (loglike, blob).
        logprior_fn: 
            function that maps one x value to log-prior.
        flow: 
            flow object with a bijection.
        scaler_cfg: 
            scaler configuration.
        scaler_masks: 
            scaler masks.
        geom_mu: 
            Student-t mean vector, shape (D,).
        geom_cov: 
            Student-t covariance matrix, shape (D, D).
        geom_nu: 
            Student-t degrees of freedom.
        n_max: 
            maximum number of inner iterations.
        n_steps: 
            value used in the stopping rule.
        proposal_scale: 
            initial proposal scale.
        condition: 
            optional conditioning array for the flow.

    Returns:
    ---------
        dict
            with updated particles, log values, counters, and proposal scale.
    """
    # all inputs to JAX arrays
    u = jnp.asarray(u)
    x = jnp.asarray(x)
    logdetj = jnp.asarray(logdetj)
    logl = jnp.asarray(logl)
    logp = jnp.asarray(logp)
    logdetj_flow = jnp.asarray(logdetj_flow)
    blobs = jnp.asarray(blobs)
    beta = jnp.asarray(beta)
    proposal_scale = jnp.asarray(proposal_scale)
    geom_mu = jnp.asarray(geom_mu)
    geom_cov = jnp.asarray(geom_cov)
    geom_nu = jnp.asarray(geom_nu)
    # define nrof walkers and parameter dimension
    n_walkers, n_dim = u.shape
    # precompute inverse covariance and Cholesky factor
    inv_cov = jnp.linalg.inv(geom_cov)
    chol_cov = jnp.linalg.cholesky(geom_cov)

    # Flow: u -> theta (batched via vmap)
    def _u2t_single(ui: Array) -> Tuple[Array, Array]:
        """
        Function maps one u vector into theta space.

        Parameters:
        -----------
            ui: 
                one particle in u-space.

        Returns:
        --------
            theta_i: 
                transformed particle in theta-space.
            logdet_i: 
                flow log determinant for that particle.
        """
        # apply forward flow to one walker
        return _flow_u_to_theta_jax(flow, ui, condition)

    # map all walkers from u-space to theta-space
    theta, logdetj_flow0 = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(u)

    # initialize running mean and adaptation values
    mu = geom_mu
    sigma0 = jnp.minimum(proposal_scale, jnp.asarray(0.99, dtype=u.dtype))
    logp2_best = jnp.mean(logl + logp)
    cnt0 = jnp.asarray(0, dtype=jnp.int32)
    i0 = jnp.asarray(0, dtype=jnp.int32)
    calls0 = jnp.asarray(0, dtype=jnp.int32)
    accept0 = jnp.asarray(0.0, dtype=u.dtype)
    done0 = jnp.asarray(False)

    # replace old flow log det with recomputed one
    logdetj_flow = logdetj_flow0
    # Use first blob shape as template for skipped likelihood calls
    blob_template = jnp.zeros_like(blobs[0])

    # helpers: Student-t form
    def _quad(diff_: Array) -> Array:
        """
        function computes the quadratic form for each row of diff_.

        Parameters:
        -----------
            diff_: 
                centered values with shape (N, D).

        Returns:
        --------
            vector with one quadratic form value per row.
        """
        # compute diff^T inv_cov diff row by row
        tmp = diff_ @ inv_cov
        return jnp.sum(tmp * diff_, axis=1)

    # skip invalid walkers
    def _prior_or_neginf(xi: Array, ok: Array) -> Array:
        """
        Function evaluates prior only when the input is valid

        Parameters:
        -----------
            xi: 
                one particle in x-space.
            ok: 
                boolean flag that says whether xi is valid.

        Returns:
        --------
            prior value, or -inf if the input is not valid.
        """
        # skip the prior if the input already failed earlier checks.
        return jax.lax.cond(
            ok,
            lambda z: logprior_fn(z),
            lambda z: jnp.asarray(-jnp.inf, dtype=xi.dtype),
            xi,
        )

    def _like_or_neginf(xi: Array, ok: Array) -> Tuple[Array, Array]:
        """
        Function evaluates likelihood only when the input is valid.

        Parameters:
        -----------
            xi: 
                one particle in x-space.
            ok: 
                boolean flag that says whether xi is valid.

        Returns:
        --------
            ll: 
                likelihood value, or -inf if the input is not valid.
            bb: 
                blob output, or a zero template if the input is not valid.
        """
        def _do(z: Array) -> Tuple[Array, Array]:
            """
            Function runs the likelihood on one valid input.

            Parameters:
            -----------
                z: one valid particle in x-space.

            Returns:
            --------
                likelihood value and blob output for that particle.
            """
            # evaluate user-provided likelihood.
            ll, bb = loglike_fn(z)
            return ll, bb

        def _skip(z: Array) -> Tuple[Array, Array]:
            """
            Function returns fallback values for an invalid input.

            Parameters:
            -----------
                z: one invalid particle in x-space.

            Returns:
            --------
                pair with -inf likelihood and a zero blob.
            """
            # keep shapes consistent when likelihood is skipped
            return jnp.asarray(-jnp.inf, dtype=xi.dtype), blob_template
        # choose between real likelihood and fallback branch.
        return jax.lax.cond(ok, _do, _skip, xi)

    
    # pack loop state into one tuple for lax.while_loop.
    carry0 = (
        key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
        mu, sigma0, logp2_best, cnt0, i0, calls0, accept0, done0
    )
    # cap adaptive proposal scale
    max_sigma_cap = jnp.minimum(jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype)),
                                jnp.asarray(0.99, dtype=u.dtype))

    def cond_fn(carry):
        """
        Function decides whether the PCN loop should continue

        Parameters:
        -----------
            carry: current loop state.

        Returns:
        --------
            boolean: True if loop should continue.
        """
        # stop when iteration limit is reached or done becomes True
        (_, _, _, _, _, _, _, _, _, _, _, _, _, i, _, _, done) = carry
        return (i < jnp.asarray(n_max, dtype=i.dtype)) & (~done)

    def body_fn(carry):
        """
        function performs one PCN update step.

        Parameters:
        -----------
            carry: current loop state.

        Returns:
        ---------
            updated loop state after one proposal step.
        """
        # unpack current loop state
        (key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
         mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry
        # move to the next inner iteration
        i1 = i + jnp.asarray(1, dtype=i.dtype)
        # split random key for all random draws in this step
        key, k_gamma, k_norm, k_unif = jax.random.split(key, 4)
        # compute current centered values and quadratic term
        diff = theta - mu
        quad = _quad(diff)
        # draw Student-t scaling factors
        a = (jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(2.0, dtype=u.dtype)
        z = jax.random.gamma(k_gamma, a, shape=(n_walkers,))  # unit scale
        s = (geom_nu + quad) / (jnp.asarray(2.0, dtype=u.dtype) * z)
        # draw Gaussian noise in geometry covariance
        eps = jax.random.normal(k_norm, shape=(n_walkers, n_dim), dtype=u.dtype)
        noise = eps @ chol_cov.T
        # build proposal in theta-space
        theta_prime = (
            mu
            + jnp.sqrt(jnp.asarray(1.0, dtype=u.dtype) - sigma * sigma) * diff
            + sigma * jnp.sqrt(s)[:, None] * noise
        )


        def _t2u_single(ti: Array) -> Tuple[Array, Array]:
            """
            Function maps one theta vector back into u-space.

            Parameters:
            -----------
                ti: one particle in theta-space.

            Returns:
            --------
                u_i: transformed particle in u-space.
                logdet_i: flow log determinant for that particle.
            """
            # apply inverse flow to one walker
            return _flow_theta_to_u_jax(flow, ti, condition)
        # map all walkers from theta-space back to u-space
        u_prime, logdetj_flow_prime = jax.vmap(_t2u_single, in_axes=0, out_axes=(0, 0))(theta_prime)

        # apply scaler inverse to move from u-space to x-space.
        x_prime, logdetj_prime = inverse_jax(u_prime, scaler_cfg, scaler_masks)

        # apply boundary handling and recompute the consistent transformed values
        x_prime_bc = apply_boundary_conditions_x_jax(x_prime, dict(scaler_cfg))
        u_prime_bc = forward_jax(x_prime_bc, scaler_cfg, scaler_masks)
        x_prime, logdetj_prime = inverse_jax(u_prime_bc, scaler_cfg, scaler_masks)

        # keep boundary-corrected u values
        u_prime = u_prime_bc

        # mark proposals that are still finite after the scaler step
        finite0 = jnp.isfinite(logdetj_prime) & jnp.all(jnp.isfinite(x_prime), axis=1)

        # evaluate prior only for finite proposals
        logp_prime = jax.vmap(_prior_or_neginf, in_axes=(0, 0), out_axes=0)(x_prime, finite0)
        finite1 = finite0 & jnp.isfinite(logp_prime)

        # evaluate likelihood only for proposals that passed the prior step
        logl_prime, blobs_prime = jax.vmap(_like_or_neginf, in_axes=(0, 0), out_axes=(0, 0))(x_prime, finite1)

        # count how many likelihood calls were actually made
        calls_inc = jnp.sum(finite1.astype(calls.dtype), dtype=calls.dtype)
        calls = calls + calls_inc

        # compute quadratic term for the proposal
        diff_prime = theta_prime - mu
        quad_prime = _quad(diff_prime)

        # build Student-t correction terms
        coef = -(jnp.asarray(n_dim, dtype=u.dtype) + geom_nu) / jnp.asarray(2.0, dtype=u.dtype)
        A = coef * jnp.log1p(quad_prime / geom_nu)
        B = coef * jnp.log1p(quad / geom_nu)

        # compute log acceptance ratio
        log_alpha = (
            (logl_prime - logl) * beta
            + (logp_prime - logp)
            + (logdetj_prime - logdetj)
            + (logdetj_flow_prime - logdetj_flow)
            - A + B
        )

        # convert to acceptance probabilities and guard against NaN
        alpha = jnp.exp(jnp.minimum(jnp.asarray(0.0, dtype=u.dtype), log_alpha))
        alpha = jnp.where(jnp.isnan(alpha), jnp.asarray(0.0, dtype=u.dtype), alpha)

        # draw uniforms and decide which walkers are accepted
        u_rand = jax.random.uniform(k_unif, shape=(n_walkers,), dtype=u.dtype)
        accept_mask = u_rand < alpha

        # apply accept/reject updates to all particle values
        theta = jnp.where(accept_mask[:, None], theta_prime, theta)
        u = jnp.where(accept_mask[:, None], u_prime, u)
        x = jnp.where(accept_mask[:, None], x_prime, x)

        logdetj = jnp.where(accept_mask, logdetj_prime, logdetj)
        logdetj_flow = jnp.where(accept_mask, logdetj_flow_prime, logdetj_flow)
        logl = jnp.where(accept_mask, logl_prime, logl)
        logp = jnp.where(accept_mask, logp_prime, logp)
        blobs = jnp.where(accept_mask.reshape((n_walkers,) + (1,) * (blobs.ndim - 1)), blobs_prime, blobs)

        accept = jnp.mean(alpha)

        # adapt proposal scale toward the target acceptance rate
        step = jnp.asarray(1.0, dtype=u.dtype) / jnp.power(jnp.asarray(i1 + 1, dtype=u.dtype), jnp.asarray(0.75, dtype=u.dtype))
        sigma = sigma + step * (accept - jnp.asarray(0.234, dtype=u.dtype))
        sigma = jnp.abs(jnp.minimum(sigma, max_sigma_cap))
        
        # update the running mean in theta-space.
        mu_step = jnp.asarray(1.0, dtype=u.dtype) / jnp.asarray(i1 + 1, dtype=u.dtype)
        mu = mu + mu_step * (jnp.mean(theta, axis=0) - mu)

        # track whether average objective improved
        logp2_new = jnp.mean(logl + logp)
        improved = logp2_new > logp2_best
        cnt = jnp.where(improved, jnp.asarray(0, dtype=cnt.dtype), cnt + jnp.asarray(1, dtype=cnt.dtype))
        logp2_best = jnp.where(improved, logp2_new, logp2_best)

        # stop when no-improvement count reaches the threshold
        thresh = jnp.asarray(n_steps, dtype=u.dtype) * jnp.power(
            (jnp.asarray(2.38, dtype=u.dtype) / jnp.sqrt(jnp.asarray(n_dim, dtype=u.dtype))) / sigma,
            jnp.asarray(2.0, dtype=u.dtype),
        )
        done = cnt.astype(u.dtype) >= thresh

        return (
            key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
            mu, sigma, logp2_best, cnt, i1, calls, accept, done
        )

    # run iterative PCN update loop
    carry_f = jax.lax.while_loop(cond_fn, body_fn, carry0)

    # unpack the final state
    (key, u, x, theta, logdetj, logdetj_flow, logl, logp, blobs,
     mu, sigma, logp2_best, cnt, i, calls, accept, done) = carry_f

    # return updated state and summary values
    return {
        "key": key,
        "u": u,
        "x": x,
        "logdetj": logdetj,
        "logdetj_flow": logdetj_flow,
        "logl": logl,
        "logp": logp,
        "blobs": blobs,
        "efficiency": sigma,
        "accept": accept,
        "steps": i,
        "calls": calls,
        "proposal_scale": sigma,
    }


