from __future__ import annotations

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import lax, random

# kinds of distribution
NORMAL = jnp.int32(0)
UNIFORM = jnp.int32(1)



def _normal_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Function computes log-density of a normal distribution.

    Parameters:
    -----------
        params: array with [loc, scale].
        x: value or array of values where the log-density is evaluated.

    Returns:
    ---------
        normal log-density at x.
    """
    # define mean and SD
    loc, scale = params[0], params[1]
    # standardize x
    z = (x - loc) / scale
    # return normal log-density
    return -0.5 * (jnp.log(2.0 * jnp.pi) + 2.0 * jnp.log(scale) + z * z)


def _uniform_logpdf(params: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Function computes log-density of a uniform distribution

    Parameters:
    -----------
        params: array with [low, high].
        x: value or array of values where the log-density is evaluated.

    Returns:
    --------
        uniform log-density at x, or -inf outside the support.
    """
    # define the interval endpoints
    low, high = params[0], params[1]
    # compute log of the interval width
    logZ = jnp.log(high - low)
    # check whether x is inside the interval
    in_support = (x >= low) & (x <= high)
    # return log-density inside interval and -inf outside it
    return jnp.where(in_support, -logZ, -jnp.inf)


def _normal_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Function draws samples from a normal distribution.

    Parameters:
    ------------
        key: JAX random key.
        params: array with [loc, scale].
        n: number of samples to draw.

    Returns:
    --------
        array of n samples from the normal distribution.
    """
    # define mean and sd
    loc, scale = params[0], params[1]
    # draw standard normal samples and rescale them
    return loc + scale * random.normal(key, shape=(n,))


def _uniform_sample(key: jax.Array, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    function draws samples from a uniform distribution.

    Parameters:
    -----------
        key: JAX random key.
        params: array with [low, high].
        n: number of samples to draw.

    Returns:
    --------
        array of n samples from the uniform distribution.
    """
    low, high = params[0], params[1]
    return random.uniform(key, shape=(n,), minval=low, maxval=high)


def _support_bounds(kind: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """
    Function returns the support bounds for one distribution.

    Parameters:
    ------------
        kind: integer code for the distribution type.
        params: parameter array for that distribution.

    Returns:
    --------
        array with [lower_bound, upper_bound].
    """
    def normal_bounds(_p):
        """
        function returns the support of a normal distribution.

        Parameters:
        -----------
            _p: unused parameter array.

        Returns:
        --------
            array with infinite bounds.
        """
        # normal distribution is supported on the whole real line
        return jnp.array([-jnp.inf, jnp.inf])

    def uniform_bounds(p):
        """
        function returns the support of a uniform distribution.

        Parameters:
        -----------
            p: parameter array with [low, high].

        Returns:
        --------
            array with the interval bounds.
        """
        # uniform distribution is supported only on [low, high]
        return jnp.array([p[0], p[1]])

    return lax.switch(kind, [normal_bounds, uniform_bounds], params)


def _logpdf_one_dim(kind: jnp.ndarray, params: jnp.ndarray, x_col: jnp.ndarray) -> jnp.ndarray:
    """
    function computes the log-density for one dimension.

    Parameters:
    -----------
        kind: integer code for the distribution type.
        params: parameter array for that distribution.
        x_col: values for one dimension.

    Returns:
    --------
        log-density values for that one dimension.
    """
    # choose correct log-density function from the distribution code
    return lax.switch(kind, [_normal_logpdf, _uniform_logpdf], params, x_col)


def _sample_one_dim(key: jax.Array, kind: jnp.ndarray, params: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Function draws samples for one dimension.

    Parameters:
    -----------
        key: JAX random key.
        kind: integer code for the distribution type.
        params: parameter array for that distribution.
        n: number of samples to draw.

    Returns:
    --------
        array of n samples for one dimension.
    """
    # choose correct sampling function from distribution
    return lax.switch(
        kind,
        [
            lambda p: _normal_sample(key, p, n),
            lambda p: _uniform_sample(key, p, n),
        ],
        params,
    )



@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Prior:
    """
    class stores a factorized prior across all dimensions.

    Parameters:
    -----------
        kinds: integer distribution codes with shape (D,).
        params: distribution parameters with shape (D, 2).

    Returns:
    ---------
        Prior object with one distribution per dimension.
    """
    kinds: jnp.ndarray   # (D,) int32
    params: jnp.ndarray  # (D, 2)

    def tree_flatten(self):
        """
        Function converts Prior object into JAX pytree parts.

        Parameters:
        -----------
            None. It uses the current Prior object.

        Returns:
        --------
            tuple with the array fields and auxiliary data.
        """
        return (self.kinds, self.params), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """
        Function rebuilds a Prior object from pytree parts.

        Parameters:
        -----------
            aux_data: auxiliary pytree data. It is unused here.
            children: tuple with the stored arrays.

        Returns:
        --------
            rebuilt Prior object.
        """
        kinds, params = children
        return cls(kinds=kinds, params=params)

    @property
    def dim(self) -> int:
        """
        Function returns the number of dimensions in the prior.

        Parameters:
        -----------
            None. It uses the current Prior object.

        Returns:
        --------
            number of dimensions as a Python int.
        """
        return int(self.kinds.shape[0])

    @staticmethod
    def create(kinds, params) -> "Prior":
        """
        Function returns the number of dimensions in the prior.

        Parameters:
        -----------
            None. It uses the current Prior object.

        Returns:
        --------
            nr of dimensions as a Python int.
        """
        # no branching; just translate into JAX arrays
        return Prior(
            kinds=jnp.asarray(kinds, dtype=jnp.int32),
            params=jnp.asarray(params),
        )

    def bounds(self) -> jnp.ndarray:
        """
        Function returns the support bounds for all dimensions.

        Parameters:
        -----------
            None. It uses the current Prior object.

        Returns:
        --------
            array with shape (D, 2) containing lower and upper bounds.
        """
        return jax.vmap(_support_bounds, in_axes=(0, 0))(self.kinds, self.params)

    # Batch logpdf: x is (N, D) -> (N,)
    def logpdf(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Function computes the batch log-density of the prior.

        Parameters:
        -----------
        x: input array with shape (N, D).

        Returns:
        --------
        array with shape (N,) containing one log-density per row.
        """
        per_dim = jax.vmap(
            _logpdf_one_dim,
            in_axes=(0, 0, 1),   # kinds (D,), params (D,2), x (N,D) into column per dim
            out_axes=1,          # (N, D)
        )(self.kinds, self.params, x)
        # sum the per-dimension terms across columns
        return jnp.sum(per_dim, axis=1)

    # Single-point logpdf: x is (D,) -> scalar
    def logpdf1(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Function computes the log-density at one point.

        Parameters:
        -----------
            x: input array with shape (D,).

        Returns:
        --------
            scalar log-density value.
        """
        return self.logpdf(x[jnp.newaxis, :])[0]

    # Batch sampling: returns (n, D)
    def sample(self, key: jax.Array, n: int) -> jnp.ndarray:
        """
        Function draws batch samples from the prior.

        Parameters:
        -----------
            key: JAX random key.
            n: number of samples to draw.

        Returns:
        --------
            array with shape (n, D).
        """
        keys = random.split(key, self.kinds.shape[0])
        # sample each dimension separately and stack the results by column
        return jax.vmap(
            lambda k, kind, p: _sample_one_dim(k, kind, p, n),
            in_axes=(0, 0, 0),
            out_axes=1,
        )(keys, self.kinds, self.params)

    # Single sample: returns (D,)
    def sample1(self, key: jax.Array) -> jnp.ndarray:
        return self.sample(key, n=1)[0]
    




    