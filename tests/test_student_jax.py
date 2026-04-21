import unittest
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.student_jax import _init_mu_sigma, fit_mvstud_jax


class StudentTest(unittest.TestCase):
    def test_init_mu_sigma(self):
        x = jnp.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=jnp.float64)
        mu, sigma = _init_mu_sigma(x)
        np.testing.assert_allclose(mu, jnp.array([1.0, 2.0]))
        np.testing.assert_allclose(sigma, sigma.T, atol=1e-12)
        assert np.all(np.diag(np.asarray(sigma)) > 0.0)

    def test_fit_mvstud(self):
        x = jnp.array(
            [[0.0, 0.0], [0.5, -0.2], [1.0, 0.3], [1.5, -0.1], [2.0, 0.1]],
            dtype=jnp.float64,
        )
        mu, sigma, nu, info = fit_mvstud_jax(x)
        assert mu.shape == (2,)
        assert sigma.shape == (2, 2)
        assert jnp.isfinite(nu)
        assert info["status"] in (0, 1, 2)


if __name__ == "__main__":
    absltest.main()
