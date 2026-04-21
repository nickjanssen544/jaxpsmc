import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.bisect_jax import bisect_jax, bisect_jax_batch


class BisectTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_root(self):
        f = lambda x: x * x - 2.0
        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(0.0, 2.0)
        np.testing.assert_allclose(root, np.sqrt(2.0), atol=1e-8)
        assert status == 0
        assert it > 0
        assert calls >= 3

    @chex.all_variants(with_pmap=False)
    def test_endpoint(self):
        f = lambda x: x - 1.0
        root, status, it, calls = self.variant(lambda a, b: bisect_jax(f, a, b))(1.0, 4.0)
        np.testing.assert_allclose(root, 1.0)
        assert status == 0
        assert it == 0
        assert calls == 2

    @chex.all_variants(with_pmap=False)
    def test_signerr(self):
        f = lambda x: x * x + 1.0
        root, status, _, _ = self.variant(lambda a, b: bisect_jax(f, a, b))(-1.0, 1.0)
        assert jnp.isnan(root)
        assert status == -1

    @chex.all_variants(with_pmap=False)
    def test_batch(self):
        def f(x, shift):
            return x - shift

        a = jnp.zeros((3,))
        b = jnp.full((3,), 3.0)
        shift = jnp.array([0.25, 1.0, 2.5])
        roots, status, it, calls = self.variant(
            lambda aa, bb, ss: bisect_jax_batch(f, aa, bb, args=(ss,))
        )(a, b, shift)
        np.testing.assert_allclose(roots, shift, atol=1e-8)
        np.testing.assert_array_equal(status, jnp.zeros((3,), dtype=jnp.int32))
        assert jnp.all(it > 0)
        assert jnp.all(calls >= 3)


if __name__ == "__main__":
    absltest.main()
