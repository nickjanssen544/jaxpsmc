import unittest
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.geometry_jax import Geometry, _cov_unweighted, _cov_weighted_aweights


class GeometryTest(unittest.TestCase):
    def test_init(self):
        g = Geometry.init(3)
        assert g.normal_mean.shape == (3,)
        assert g.normal_cov.shape == (3, 3)
        assert g.t_mean.shape == (3,)
        assert g.t_cov.shape == (3, 3)
        assert g.t_nu.shape == ()

    def test_cov_unweighted(self):
        x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        mu, cov = _cov_unweighted(x, jitter=jnp.array(1e-9))
        np.testing.assert_allclose(mu, jnp.array([2.0, 3.0]))
        np.testing.assert_allclose(cov, cov.T, atol=1e-12)
        assert np.all(np.diag(np.asarray(cov)) > 0.0)

    def test_bad_weights_fall_back_to_unweighted(self):
        x = jnp.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        w = jnp.array([jnp.nan, 1.0, 2.0])
        mu_w, cov_w = _cov_weighted_aweights(x, w, jitter=jnp.array(1e-9))
        mu_u, cov_u = _cov_unweighted(x, jitter=jnp.array(1e-9))
        np.testing.assert_allclose(mu_w, mu_u, atol=1e-12)
        np.testing.assert_allclose(cov_w, cov_u, atol=1e-12)


if __name__ == "__main__":
    absltest.main()
