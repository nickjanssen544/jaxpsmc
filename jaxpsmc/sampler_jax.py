from __future__ import annotations
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

# helper modules used by the sampler
from .bisect_jax import *
from .geometry_jax import *
from .input_validation_jax import *
from .particles_jax import *
from .pcn_jax import *
from .prior_jax import *
from .sampler_helper_jax import *
from .scaler_jax import *
from .student_jax import *
from .tools_jax import *





Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityBijectionJAX:
    """
    Function-like class stores an identity bijection with zero log-determinant.

    Parameters:
    -----------
        None.

    Returns:
    --------
        object that matches the flow bijection interface.
    """

    def tree_flatten(self):
        """
        Function converts the object into JAX pytree parts.

        Parameters:
        -----------
            None.

        Returns:
        --------
            empty children tuple and auxiliary data.
        """
        # This object has no array fields.
        return (), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Function rebuilds the object from JAX pytree parts.

        Parameters:
        -----------
            aux: auxiliary pytree data.
            children: pytree children.

        Returns:
        --------
            rebuilt IdentityBijectionJAX object.
        """
        return cls()

    def transform_and_log_det(self, u: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Function maps u to itself and returns zero log-determinant.

        Parameters:
        -----------
            u: input array in latent space.
            condition: optional conditioning input.

        Returns:
        --------
            unchanged array and zero log-determinant.
        """
        # convert input to a JAX array
        u = jnp.asarray(u)
        return u, jnp.zeros(u.shape[:-1], dtype=u.dtype)

    def inverse_and_log_det(self, theta: Array, condition: Optional[Array] = None) -> Tuple[Array, Array]:
        """
        Function maps theta to itself and returns zero log-determinant.

        Parameters:
        -----------
            theta: input array in transformed space.
            condition: optional conditioning input.

        Returns:
        --------
            unchanged array and zero log-determinant.
        """
        # convert input to a JAX array
        theta = jnp.asarray(theta)
        return theta, jnp.zeros(theta.shape[:-1], dtype=theta.dtype)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class IdentityFlowJAX:
    """
    Function-like class stores a simple flow object with an identity bijection.

    Parameters:
    -----------
        dim: dimension of the latent space.

    Returns:
    --------
        object that behaves like a flow for the sampler.
    """

    dim: int

    def tree_flatten(self):
        """
        Function converts the object into JAX pytree parts.

        Parameters:
        -----------
            None.

        Returns:
        --------
            empty children tuple and auxiliary data with the dimension.
        """
        # store dimension as auxiliary data
        return (), (self.dim,)

    @classmethod
    def tree_unflatten(cls, aux, children):
        """
        Function rebuilds the object from JAX pytree parts.

        Parameters:
        -----------
            aux: auxiliary pytree data that stores the dimension.
            children: pytree children.

        Returns:
        --------
            rebuilt IdentityFlowJAX object.
        """
        # define stored dimension and rebuild object
        (dim,) = aux
        return cls(dim=dim)

    @property
    def bijection(self) -> IdentityBijectionJAX:
        """
        Function returns the identity bijection used by the flow.

        Parameters:
        -----------
            None.

        Returns:
        --------
            IdentityBijectionJAX object.
        """
        # return a fresh identity bijection
        return IdentityBijectionJAX()

    def fit(self, *args, **kwargs):
        """
        Function keeps the same flow object when fit is called.

        Parameters:
        -----------
            *args: unused positional inputs.
            **kwargs: unused keyword inputs.

        Returns:
        --------
            current IdentityFlowJAX object.
        """
        # flow has no trainable state, so return itself
        return self

    def sample(self, key: Array, n: int, condition: Optional[Array] = None) -> Array:
        """
        Function draws standard normal samples from the identity flow.

        Parameters:
        -----------
            key: JAX random key.
            n: number of samples to draw.
            condition: optional conditioning input.

        Returns:
        --------
            array with shape (n, dim).
        """
        # draw standard normal samples in the latent space
        return jax.random.normal(key, (n, self.dim))






##############################################################
# 1. CONFIGURATION HELPERS
##############################################################


def _metric_code(metric: str) -> jnp.int32:
    """
    Function converts the metric name into its integer code.

    Parameters:
    -----------
        metric: metric name, expected to be "ess" or "uss".

    Returns:
    --------
        integer code for the selected metric.
    """
    # normalize input string before checking it
    metric_l = str(metric).lower()

    # map metric name to internal code
    if metric_l == "ess":
        return METRIC_ESS
    if metric_l == "uss":
        return METRIC_USS
    raise ValueError("metric must be 'ess' or 'uss'.")


def _resample_code(resample: str) -> jnp.int32:
    """
    Function converts the resampling name into its integer code.

    Parameters:
    -----------
        resample: resampling name, expected to be "mult" or "syst".

    Returns:
    --------
        integer code for the selected resampling method.
    """
    # normalize input string before checking it
    res_l = str(resample).lower()

    # map method name to internal code
    if res_l == "mult":
        return jnp.int32(0)
    if res_l == "syst":
        return jnp.int32(1)
    raise ValueError("resample must be 'mult' or 'syst'.")


@dataclass(frozen=True)
class SamplerConfigJAX:
    """
    Function-like class stores sampler settings.

    Parameters:
    -----------
        n_dim: problem dimension.
        n_effective: target effective sample size.
        n_active: number of active particles.
        n_prior: number of prior samples used in warmup.
        n_total: total stopping target for the sampler.
        n_steps: stopping value used by the mutation step.
        n_max_steps: maximum number of outer sampler steps.
        proposal_scale: initial proposal scale.
        keep_max: maximum number of particles kept after trimming.
        trim_ess: ESS ratio used when trimming weights.
        bins: number of bins used in trimming.
        bisect_steps: number of bisection steps in reweighting.
        preconditioned: whether to use the preconditioned mutation step.
        dynamic: whether to update the effective target dynamically.
        metric: metric name, either "ess" or "uss".
        resample: resampling name, either "mult" or "syst".
        transform: scaler transform name.
        periodic: optional periodic coordinate indices.
        reflective: optional reflective coordinate indices.
        blob_dim: size of the blob output.
        enable_flow_evidence: whether to enable flow-based evidence code.

    Returns:
    --------
        SamplerConfigJAX object.
    """
    # dimensions
    n_dim: int
    n_effective: int = 512
    n_active: int = 256
    n_prior: int = 512

    # SMC termination
    n_total: int = 4096

    # MCMC kernel
    n_steps: int = 8
    n_max_steps: int = 80
    proposal_scale: float = 0.0     # if 0 then set to 2.38/sqrt(D)

    # reweight and trim
    keep_max: int = 4096
    trim_ess: float = 0.99
    bins: int = 1000
    bisect_steps: int = 32

    # resampling options: ess or uss and syst or mult
    preconditioned: bool = True
    dynamic: bool = True
    metric: str = "ess"             # "ess" or "uss" 
    resample: str = "mult"          # "mult" or "syst" 

    # scaler options
    transform: str = "probit"       # "probit" or "logit"
    periodic: Optional[jnp.ndarray] = None
    reflective: Optional[jnp.ndarray] = None

    # initiate blobs
    blob_dim: int = 0

    # evidence option
    enable_flow_evidence: bool = False

    def __post_init__(self):
        """
        Function validates the main configuration values.

        Parameters:
        -----------
            None.

        Returns:
        --------
            None.
        """
        # require positive dimensions and particle counts
        if self.n_active <= 0 or self.n_effective <= 0 or self.n_dim <= 0:
            raise ValueError("n_dim, n_active, n_effective must be positive.")
        # warmup samples must split evenly into active-particle batches
        if self.n_prior % self.n_active != 0:
            raise ValueError("n_prior must be a multiple of n_active for warmup batching.")
        # keep limit must also be positive.
        if self.keep_max <= 0:
            raise ValueError("keep_max must be positive.")


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RunOutputJAX:
    """
    Function-like class stores the final sampler outputs as a JAX pytree.

    Parameters:
    -----------
        state: full particle history.
        logz: final evidence estimate.
        logz_err: error estimate for logz.

    Returns:
    --------
        RunOutputJAX object.
    """
    state: ParticlesState
    logz: Array
    logz_err: Array

    def tree_flatten(self):
        """
        Function converts the object into JAX pytree parts.

        Parameters:
        -----------
            None.

        Returns:
        --------
            tuple with stored arrays and auxiliary data.
        """
        # ParticlesState is already a pytree, so return it directly
        return (self.state, self.logz, self.logz_err), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Function rebuilds the object from JAX pytree parts.

        Parameters:
        -----------
            aux_data: auxiliary pytree data.
            children: tuple with stored fields.

        Returns:
        --------
            rebuilt RunOutputJAX object.
        """
        # read stored fields and rebuild object
        state, logz, logz_err = children
        return cls(state=state, logz=logz, logz_err=logz_err)


class SamplerJAX:
    """
    Function-like wrapper stores the objects needed to run the sampler.

    Parameters:
    -----------
        prior: prior object.
        loglike_single_fn: single-point likelihood function.
        cfg: sampler configuration.
        flow: optional flow object.

    Returns:
    --------
        SamplerJAX object.
    """

    def __init__(
        self,
        prior: Prior,
        loglike_single_fn: Callable[[Array], Any],
        cfg: SamplerConfigJAX,
        *,
        flow: Optional[Any] = None,
    ):
        """
        Function initializes the sampler wrapper.

        Parameters:
        -----------
            prior: prior object.
            loglike_single_fn: single-point likelihood function.
            cfg: sampler configuration.
            flow: optional flow object.

        Returns:
        --------
            None.
        """
        # store main objects used by sampler
        self.prior = prior
        self.cfg = cfg
        self.flow = IdentityFlowJAX(cfg.n_dim) if flow is None else flow
        # build jitted run function once during initialization
        self._run_fn = make_run_fn(prior=prior, loglike_single_fn=loglike_single_fn, cfg=cfg, flow=self.flow)

    def run(self, key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        """
        Function runs the sampler.

        Parameters:
        -----------
            key: JAX random key.
            n_total: optional stopping target that overrides the config value.

        Returns:
        --------
            RunOutputJAX object.
        """
        # forward call to stored jitted run function
        return self._run_fn(key, n_total=n_total)







##############################################################
# 2. CORE JAX RUN LOOP 
##############################################################

def _replace_inf_rows(
    key: Array,
    x: Array,
    u: Array,
    logdetj: Array,
    logp: Array,
    logl: Array,
    blobs: Array,
) -> Tuple[Array, Array, Array, Array, Array, Array, Array]:
    """
    Function replaces rows with infinite likelihood values by copying finite rows.

    Parameters:
    -----------
        key: JAX random key.
        x: particle values in x-space.
        u: particle values in u-space.
        logdetj: scaler log-determinant values.
        logp: log-prior values.
        logl: log-likelihood values.
        blobs: blob outputs.

    Returns:
    --------
        updated key and updated particle arrays.
    """
    # define number of rows
    n = x.shape[0]
    # mark rows with infinite likelihood values
    inf_mask = jnp.isinf(logl)
    finite_mask = ~inf_mask

    # build sampling probabilities from finite rows only
    probs = finite_mask.astype(x.dtype)
    psum = jnp.sum(probs)
    probs = probs / jnp.where(psum > 0, psum, jnp.asarray(1.0, x.dtype))
    logits = jnp.where(probs > 0, jnp.log(probs), -jnp.inf)

    # draw replacement rows for invalid entries
    key, sub = jax.random.split(key)
    idx_rep = jax.random.categorical(sub, logits, shape=(n,), axis=0).astype(jnp.int32)
    idx_self = jnp.arange(n, dtype=jnp.int32)
    idx = jnp.where(inf_mask, idx_rep, idx_self)

    # gather updated arrays
    x2 = jnp.take(x, idx, axis=0)
    u2 = jnp.take(u, idx, axis=0)
    logdetj2 = jnp.take(logdetj, idx, axis=0)
    logp2 = jnp.take(logp, idx, axis=0)
    logl2 = jnp.take(logl, idx, axis=0)
    blobs2 = jnp.take(blobs, idx, axis=0)
    return key, x2, u2, logdetj2, logp2, logl2, blobs2


def _build_step_from_particles(
    *,
    u: Array,
    x: Array,
    logdetj: Array,
    logl: Array,
    logp: Array,
    blobs: Array,
    iter_idx: Array,
    beta: Array,
    logz: Array,
    calls: Array,
    steps: Array,
    efficiency: Array,
    ess: Array,
    accept: Array,
) -> ParticlesStep:
    """
    Function builds one ParticlesStep object from particle arrays and summary values.

    Parameters:
    -----------
        u: particle values in u-space.
        x: particle values in x-space.
        logdetj: scaler log-determinant values.
        logl: log-likelihood values.
        logp: log-prior values.
        blobs: blob outputs.
        iter_idx: current iteration index.
        beta: current beta value.
        logz: current logz value.
        calls: current number of likelihood calls.
        steps: number of mutation steps.
        efficiency: efficiency value.
        ess: effective sample size value.
        accept: acceptance value.

    Returns:
    --------
        ParticlesStep object.
    """
    # store a placeholder log-weight vector for the history record
    logw = jnp.zeros_like(logl)
    # buils ParticlesStep object
    return ParticlesStep(
        u=u,
        x=x,
        logdetj=logdetj,
        logl=logl,
        logp=logp,
        logw=logw,
        blobs=blobs,
        iter=iter_idx.astype(jnp.int32),
        logz=logz,
        calls=calls,
        steps=steps,
        efficiency=efficiency,
        ess=ess,
        accept=accept,
        beta=beta,
    )


def make_run_fn(
    *,
    prior: Prior,
    loglike_single_fn: Callable[[Array], Tuple[Array, Array]],
    cfg: SamplerConfigJAX,
    flow: Optional[Any] = None,
) -> Callable[[Array], RunOutputJAX]:
    """
    Function builds the jitted sampler run function.

    Parameters:
    -----------
        prior: prior object.
        loglike_single_fn: single-point likelihood function.
        cfg: sampler configuration.
        flow: optional flow object.

    Returns:
    --------
        function that runs the sampler from a random key.
    """
    # useidentity flow when no flow object is given

    flow_obj = IdentityFlowJAX(cfg.n_dim) if flow is None else flow

    # read fixed blob size from config
    blob_dim = int(cfg.blob_dim)

    def loglike_wrapped(x: Array) -> Tuple[Array, Array]:
        """
        Function converts the user likelihood into a fixed output shape.

        Parameters:
        -----------
            x: one point with shape (D,).

        Returns:
        --------
            scalar log-likelihood and blob vector with shape (blob_dim,).
        """
        # call user-provided likelihood function
        out = loglike_single_fn(x)

        # accept either a scalar return or a pair with a blob
        if isinstance(out, tuple) and len(out) == 2:
            ll, blob = out
        else:
            ll, blob = out, jnp.zeros((blob_dim,), dtype=jnp.result_type(out, jnp.float64))

        # build a blob vector with configured size
        if blob_dim == 0:
            blob_vec = jnp.zeros((0,), dtype=jnp.result_type(ll, jnp.float64))
        else:
            blob_vec = jnp.asarray(blob).reshape((blob_dim,))
        return jnp.asarray(ll), blob_vec
    
    # convert string options into integer codes
    metric_code = _metric_code(cfg.metric)
    res_code = _resample_code(cfg.resample)

    # build periodic and reflective index arrays
    periodic = jnp.asarray(cfg.periodic if cfg.periodic is not None else jnp.zeros((0,), dtype=jnp.int64))
    reflective = jnp.asarray(cfg.reflective if cfg.reflective is not None else jnp.zeros((0,), dtype=jnp.int64))

    # build scaler configuration from prior bounds
    bounds = prior.bounds()
    scaler_cfg0 = init_bounds_config_jax(
        cfg.n_dim,
        bounds=bounds,
        periodic=periodic,
        reflective=reflective,
        transform=cfg.transform,
        scale=True,
        diagonal=True,
    )
    scaler_masks = masks_jax(scaler_cfg0["low"], scaler_cfg0["high"])

    # precompute dynamic ratio used by reweight step
    w_ones = jnp.ones((cfg.n_effective,), dtype=jnp.float64)
    dyn_ratio = (unique_sample_size_jax(w_ones, k=cfg.n_active) / jnp.asarray(cfg.n_active, jnp.float64)).astype(jnp.float64)

    # choose initial proposal scale
    prop_scale = (
        (2.38 / (cfg.n_dim ** 0.5))
        if (cfg.proposal_scale is None or cfg.proposal_scale == 0.0)
        else float(cfg.proposal_scale)
    )

    # preallocate particle history slots for warmup and SMC
    max_steps_total = int((cfg.n_prior // cfg.n_active) + cfg.n_max_steps)

    @partial(
        jax.jit,
        static_argnames=(
            "n_active",
            "n_prior",
            "n_steps",
            "n_max_steps",
            "keep_max",
            "bins",
            "bisect_steps",
            "trim_ess",
            "blob_dim",
        ),
    )
    def _run(
        key: Array,
        n_total_dyn: Array,
        *,
        n_active: int,
        n_prior: int,
        n_steps: int,
        n_max_steps: int,
        keep_max: int,
        bins: int,
        bisect_steps: int,
        trim_ess: float,
        blob_dim: int,
    ) -> RunOutputJAX:
        """
        Function runs the full sampler with fixed static settings.

        Parameters:
        -----------
            key: JAX random key.
            n_total_dyn: stopping target as a JAX scalar.
            n_active: number of active particles.
            n_prior: number of prior samples used in warmup.
            n_steps: stopping value used by mutation.
            n_max_steps: maximum number of outer sampler steps.
            keep_max: maximum number of kept particles after trimming.
            bins: number of trimming bins.
            bisect_steps: number of bisection steps.
            trim_ess: ESS ratio used by trimming.
            blob_dim: blob output size.

        Returns:
        --------
            RunOutputJAX object.
        """
        # convert key and choose dtype
        key = jnp.asarray(key)
        dtype = jnp.result_type(prior.params, jnp.float64)

        # (i) sample prior points used to fit the scaler
        key, k_prior = jax.random.split(key)
        prior_samples = prior.sample(k_prior, n_prior).astype(dtype)  # (n_prior, D)

        # (ii) fit scaler on the prior samples
        scaler_cfg = fit_jax(prior_samples, scaler_cfg0, scaler_masks)

        # (iii) create particle-history buffers
        state = init_particles_state_jax(
            max_steps=max_steps_total,
            n_particles=n_active,
            n_dim=cfg.n_dim,
            blob_dim=blob_dim,
            dtype=dtype,
        )

        #  compute number of warmup batches
        n_warm = n_prior // n_active
        geom0 = Geometry.init(cfg.n_dim, dtype=dtype)

        # build initial scalar values used during warmup
        calls0 = jnp.asarray(0, dtype=jnp.int32)
        beta0 = jnp.asarray(0.0, dtype=dtype)
        logz0 = jnp.asarray(0.0, dtype=dtype)
        ess0 = jnp.asarray(cfg.n_effective, dtype=dtype)
        accept0 = jnp.asarray(1.0, dtype=dtype)
        steps0 = jnp.asarray(1, dtype=jnp.int32)
        eff0 = jnp.asarray(1.0, dtype=dtype)

        def warm_body(carry, i):
            """
            Function processes one warmup batch of prior samples.

            Parameters:
            -----------
                carry: current key, particle state, and call count.
                i: warmup batch index.

            Returns:
            --------
                updated carry and no scan output.
            """
            # unpack current warmup state
            key_c, state_c, calls_c = carry

            # slice out one batch of prior samples
            start = (i * n_active)
            x = lax.dynamic_slice_in_dim(prior_samples, start_index=start, slice_size=n_active, axis=0)

            # map x into u-space and recompute log-determinant
            u = forward_jax(x, scaler_cfg, scaler_masks)
            _x_back, logdetj = inverse_jax(u, scaler_cfg, scaler_masks)

            # evaluate prior and likelihood values for batch
            logp = prior.logpdf(x)
            logl, blobs = jax.vmap(loglike_wrapped, in_axes=0, out_axes=(0, 0))(x)
            blobs = blobs.astype(dtype)

            # count likelihood calls from this batch
            calls_c = calls_c + jnp.asarray(n_active, dtype=calls_c.dtype)

            # replace rows with infinite likelihood values
            key_c, x, u, logdetj, logp, logl, blobs = _replace_inf_rows(
                key_c, x, u, logdetj, logp, logl, blobs
            )

            # build and record one history step
            step = _build_step_from_particles(
                u=u,
                x=x,
                logdetj=logdetj,
                logl=logl,
                logp=logp,
                blobs=blobs,
                iter_idx=state_c.t,
                beta=beta0,
                logz=logz0,
                calls=calls_c.astype(dtype),
                steps=steps0.astype(dtype),
                efficiency=eff0,
                ess=ess0,
                accept=accept0,
            )
            state_c = record_step_jax(state_c, step)
            return (key_c, state_c, calls_c), None

        # run warmup over all prior batches
        (key, state, calls_w), _ = lax.scan(
            warm_body,
            (key, state, calls0),
            xs=jnp.arange(n_warm, dtype=jnp.int32),
        )

        # initialize values used by outer SMC loop
        n_eff_c = jnp.asarray(cfg.n_effective, dtype=jnp.int32)
        iter0 = jnp.asarray(0, dtype=jnp.int32)

        # initiate the most recent warmup particles
        last_u = lax.dynamic_index_in_dim(state.u, state.t - 1, axis=0, keepdims=False)
        last_x = lax.dynamic_index_in_dim(state.x, state.t - 1, axis=0, keepdims=False)
        last_logdetj = lax.dynamic_index_in_dim(state.logdetj, state.t - 1, axis=0, keepdims=False)
        last_logl = lax.dynamic_index_in_dim(state.logl, state.t - 1, axis=0, keepdims=False)
        last_logp = lax.dynamic_index_in_dim(state.logp, state.t - 1, axis=0, keepdims=False)
        last_blobs = lax.dynamic_index_in_dim(state.blobs, state.t - 1, axis=0, keepdims=False)

        # build initial current-particle dictionary
        current_particles0: Dict[str, Array] = {
            "u": last_u,
            "x": last_x,
            "logdetj": last_logdetj,
            "logl": last_logl,
            "logp": last_logp,
            "logdetj_flow": jnp.zeros((n_active,), dtype=dtype),
            "blobs": last_blobs,
            "beta": beta0,
            "calls": calls_w,
            "proposal_scale": jnp.asarray(prop_scale, dtype=dtype),
            # IMPORTANT: 
            # PyTree structure is fixed across SMC while_loop
            # _mutate() always returns scalar diagnostics, so 
            # include them in the initial carry as well
            "efficiency": jnp.asarray(1.0, dtype=dtype),
            "accept": jnp.asarray(1.0, dtype=dtype),
            "steps": jnp.asarray(0, dtype=jnp.int32),
        }

        def _u2t_single(ui: Array) -> Tuple[Array, Array]:
            """
            Function maps one latent vector into theta-space.

            Parameters:
            -----------
                ui: one latent vector.

            Returns:
            --------
                transformed vector and flow log-determinant.
            """
            # use flow bijection to one latent vector
            theta, logdet = flow_obj.bijection.transform_and_log_det(ui, None)
            return theta, logdet

        # build an initial geometry object from warmup particles
        theta0, _ = jax.vmap(_u2t_single, in_axes=0, out_axes=(0, 0))(current_particles0["u"])
        w0 = jnp.full((n_active,), jnp.asarray(1.0, dtype) / jnp.asarray(n_active, dtype), dtype=dtype)
        geom, key, _ = geometry_fit_jax(geom0, theta0, w0, use_weights=jnp.asarray(False), key=key)

        # convert loop settings to JAX arrays
        n_total = jnp.asarray(n_total_dyn, dtype=dtype)
        metric_id = jnp.asarray(metric_code, dtype=jnp.int32)
        n_active_i32 = jnp.asarray(n_active, dtype=jnp.int32)
        res_code_i32 = jnp.asarray(res_code, dtype=jnp.int32)
        dyn_ratio_arr = jnp.asarray(dyn_ratio, dtype=dtype)
        use_pcn = jnp.asarray(cfg.preconditioned)
        dynamic = jnp.asarray(cfg.dynamic)

        def cond_fn(carry):
            """
            Function checks whether the outer SMC loop should continue.

            Parameters:
            -----------
                carry: current loop state.

            Returns:
            --------
                boolean value that is True when another step is needed.
            """
            # unpack values needed by the stop rule
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry
            
            # continue while sampler has not met stop rule
            not_done = not_termination_jax(
                state_c,
                beta_current=cur_c["beta"],
                n_total=n_total,
                metric_code=metric_id,
                n_active=n_active_i32,
            )
            within_cap = it < jnp.asarray(n_max_steps, dtype=it.dtype)
            return not_done & within_cap

        def body_fn(carry):
            """
            Function runs one outer SMC step.

            Parameters:
            -----------
                carry: current loop state.

            Returns:
            --------
                updated loop state.
            """
            # unpack current loop state
            key_c, state_c, cur_c, geom_c, n_eff_c2, it = carry

            # reweight history and keep the highest weight particles
            cur_rw, n_eff_new, stats = reweight_step_jax(
                state_c,
                n_eff_c2,
                metric_id,
                dynamic,
                n_active_i32,
                dyn_ratio_arr,
                bins=bins,
                bisect_steps=bisect_steps,
                keep_max=keep_max,
                trim_ess=trim_ess,
            )

            def _u2t_keep(ui: Array) -> Tuple[Array, Array]:
                """
                Function maps one kept latent vector into theta-space.

                Parameters:
                -----------
                    ui: one latent vector.

                Returns:
                --------
                    transformed vector and flow log-determinant.
                """
                # apply flow bijection to one kept particle
                th, ld = flow_obj.bijection.transform_and_log_det(ui, None)
                return th, ld

            # update geometry using kept weighted particles            
            theta_keep, _ = jax.vmap(_u2t_keep, in_axes=0, out_axes=(0, 0))(cur_rw["u"])
            geom_new, key_c, _ = geometry_fit_jax(
                geom_c,
                theta_keep,
                cur_rw["weights"],
                use_weights=jnp.asarray(True),
                key=key_c,
            )

            # resample kept particles down to active set
            rs_out, _status, key_c = resample_particles_jax(
                cur_rw,
                key=key_c,
                n_active=n_active,
                method_code=res_code_i32,
                reset_weights=True,
            )

            # build input dictionary expected by mutation step
            cur_for_mut = {
                "u": rs_out["u"],
                "x": rs_out["x"],
                "logdetj": rs_out["logdetj"],
                "logl": rs_out["logl"],
                "logp": rs_out["logp"],
                "logdetj_flow": jnp.zeros((n_active,), dtype=dtype),
                "blobs": rs_out["blobs"],
                "beta": cur_rw["beta"],
                "calls": cur_c["calls"],
                "proposal_scale": cur_c["proposal_scale"],
            }

            # run mutation step
            key_c, mutated, info = mutate(
                key_c,
                cur_for_mut,
                use_preconditioned_pcn=use_pcn,
                loglike_single_fn=loglike_wrapped,
                logprior_fn=prior.logpdf1,
                flow=flow_obj,
                scaler_cfg=scaler_cfg,
                scaler_masks=scaler_masks,
                geom_mu=geom_new.t_mean,
                geom_cov=geom_new.t_cov,
                geom_nu=geom_new.t_nu,
                n_max=n_max_steps,
                n_steps=n_steps,
                condition=None,
            )

            # record new mutated particles into history
            step = _build_step_from_particles(
                u=mutated["u"],
                x=mutated["x"],
                logdetj=mutated["logdetj"],
                logl=mutated["logl"],
                logp=mutated["logp"],
                blobs=mutated["blobs"],
                iter_idx=state_c.t,
                beta=cur_rw["beta"],
                logz=cur_rw["logz"],
                calls=mutated["calls"].astype(dtype),
                steps=mutated["steps"].astype(dtype),
                efficiency=mutated["efficiency"],
                ess=stats["ess"],
                accept=mutated["accept"],
            )
            state_c = record_step_jax(state_c, step)

            # define next current particle dictionary
            cur_next = {
                **mutated,
                "beta": cur_rw["beta"],
                "calls": mutated["calls"],
                "proposal_scale": mutated["proposal_scale"],
            }

            return (key_c, state_c, cur_next, geom_new, n_eff_new, it + jnp.int32(1))

        # run outer SMC loop
        key, state, cur, geom, n_eff_c, itf = lax.while_loop(
            cond_fn,
            body_fn,
            (key, state, current_particles0, geom, n_eff_c, iter0),
        )

        # compute final log-evidence from stored history
        _logw_flat, logz_final, _mask = compute_logw_and_logz_jax(
            state,
            beta_final=jnp.asarray(1.0, dtype=dtype),
            normalize=False,
        )
        logz_err = jnp.asarray(jnp.nan, dtype=dtype)

        return RunOutputJAX(state=state, logz=logz_final, logz_err=logz_err)

    def run(key: Array, n_total: Optional[int] = None) -> RunOutputJAX:
        """
        Function runs the sampler with the captured static objects.

        Parameters:
        -----------
            key: JAX random key.
            n_total: optional stopping target that overrides the config value.

        Returns:
        --------
            RunOutputJAX object.
        """
        # use stopping or fall back to config value
        n_total_use = cfg.n_total if n_total is None else int(n_total)
        
        # run jitted core run function
        return _run(
            key,
            n_total_dyn=jnp.asarray(n_total_use, dtype=jnp.float64),
            n_active=cfg.n_active,
            n_prior=cfg.n_prior,
            n_steps=cfg.n_steps,
            n_max_steps=cfg.n_max_steps,
            keep_max=cfg.keep_max,
            bins=cfg.bins,
            bisect_steps=cfg.bisect_steps,
            trim_ess=cfg.trim_ess,
            blob_dim=cfg.blob_dim,
        )

    return run




