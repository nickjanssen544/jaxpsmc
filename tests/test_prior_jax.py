import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.prior_jax import NORMAL, UNIFORM, Prior


class PriorTest(chex.TestCase):
    def test_create_and_bounds(self):
        prior = Prior.create([NORMAL, UNIFORM], [[0.0, 1.0], [-2.0, 3.0]])
        assert prior.dim == 2
        bounds = prior.bounds()
        assert jnp.isneginf(bounds[0, 0])
        assert jnp.isposinf(bounds[0, 1])
        np.testing.assert_allclose(bounds[1], jnp.array([-2.0, 3.0]))

    @chex.all_variants(with_pmap=False)
    def test_logpdf(self):
        prior = Prior.create([NORMAL, UNIFORM], [[0.0, 1.0], [0.0, 2.0]])
        x = jnp.array([[0.0, 1.0], [1.0, 3.0]])
        logp = self.variant(prior.logpdf)(x)
        expected0 = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(2.0)
        np.testing.assert_allclose(logp[0], expected0)
        assert jnp.isneginf(logp[1])
        np.testing.assert_allclose(prior.logpdf1(x[0]), logp[0])

    @chex.all_variants(with_pmap=False)
    def test_sample(self):
        prior = Prior.create([NORMAL, UNIFORM], [[0.0, 1.0], [-1.0, 1.0]])
        samples = self.variant(lambda k: prior.sample(k, 2000))(jax.random.key(2))
        assert samples.shape == (2000, 2)
        assert jnp.all(samples[:, 1] >= -1.0)
        assert jnp.all(samples[:, 1] <= 1.0)
        np.testing.assert_allclose(jnp.mean(samples[:, 0]), 0.0, atol=0.1)


if __name__ == "__main__":
    absltest.main()
