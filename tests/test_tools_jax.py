import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.tools_jax import (
    compute_ess_jax,
    effective_sample_size_jax,
    increment_logz_jax,
    systematic_resample_jax,
    systematic_resample_jax_size,
    trim_weights_jax,
    unique_sample_size_jax,
)


class ToolsTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_trim_weights(self):
        w = jnp.array([0.6, 0.3, 0.1])
        mask, w_trim, thr, ratio, _ = self.variant(
            lambda z: trim_weights_jax(jnp.arange(z.shape[0]), z, ess=0.9, bins=64)
        )(w)
        assert mask.shape == (3,)
        assert w_trim.shape == (3,)
        np.testing.assert_allclose(jnp.sum(w_trim), 1.0)
        assert jnp.isfinite(thr)
        assert ratio >= 0.9

    @chex.all_variants(with_pmap=False)
    def test_ess_and_uss(self):
        w = jnp.array([0.5, 0.5])
        ess = self.variant(effective_sample_size_jax)(w)
        uss = self.variant(lambda z: unique_sample_size_jax(z, k=2))(w)
        np.testing.assert_allclose(ess, 2.0)
        np.testing.assert_allclose(uss, 1.5)

    @chex.all_variants(with_pmap=False)
    def test_compute_ess_and_logz(self):
        logw = jnp.array([0.0, 0.0])
        ess = self.variant(compute_ess_jax)(logw)
        inc = self.variant(increment_logz_jax)(logw)
        np.testing.assert_allclose(ess, 1.0)
        np.testing.assert_allclose(inc, jnp.log(2.0))

    @chex.all_variants(with_pmap=False)
    def test_systematic(self):
        w = jnp.array([0.7, 0.2, 0.1])
        idx, status, _ = self.variant(lambda z: systematic_resample_jax(z, key=jax.random.key(0)))(w)
        assert idx.shape == (3,)
        assert jnp.all((idx >= 0) & (idx < 3))
        assert status == 0

    @chex.all_variants(with_pmap=False)
    def test_systematic_size_and_bad_weights(self):
        bad = jnp.array([jnp.nan, 1.0])
        idx, status, _ = self.variant(
            lambda z: systematic_resample_jax_size(z, key=jax.random.key(1), size=4)
        )(bad)
        np.testing.assert_array_equal(idx, jnp.full((4,), -1, dtype=jnp.int32))
        assert status == -3


if __name__ == "__main__":
    absltest.main()
