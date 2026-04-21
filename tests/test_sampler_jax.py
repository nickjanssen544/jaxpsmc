import unittest
import jax
import jax.numpy as jnp
from absl.testing import absltest

from jaxpsmc.prior_jax import NORMAL, Prior
from jaxpsmc.sampler_jax import (
    IdentityBijectionJAX,
    IdentityFlowJAX,
    SamplerConfigJAX,
    SamplerJAX,
    _build_step_from_particles,
    _metric_code,
    _replace_inf_rows,
    _resample_code,
)


class SamplerTest(unittest.TestCase):
    def test_identity_flow(self):
        flow = IdentityFlowJAX(2)
        bij = flow.bijection
        u = jnp.array([[0.0, 1.0], [2.0, 3.0]])
        x, ld1 = bij.transform_and_log_det(u)
        u2, ld2 = bij.inverse_and_log_det(x)
        import numpy as np
        np.testing.assert_allclose(x, u)
        np.testing.assert_allclose(u2, u)
        assert ld1.shape == (2,)
        assert ld2.shape == (2,)

    def test_codes_and_config(self):
        assert _metric_code("ess") == 0
        assert _metric_code("uss") == 1
        assert _resample_code("mult") == 0
        assert _resample_code("syst") == 1
        cfg = SamplerConfigJAX(n_dim=2, n_active=4, n_effective=2, n_prior=8, keep_max=4)
        assert cfg.n_dim == 2

    def test_replace_inf_rows(self):
        key, x, u, logdetj, logp, logl, blobs = _replace_inf_rows(
            jax.random.key(0),
            jnp.array([[0.0], [1.0], [2.0]]),
            jnp.array([[0.0], [1.0], [2.0]]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, 0.0, 0.0]),
            jnp.array([0.0, jnp.inf, 1.0]),
            jnp.zeros((3, 0)),
        )
        assert jnp.isfinite(logl).all()
        assert x.shape == (3, 1)
        assert u.shape == (3, 1)
        assert blobs.shape == (3, 0)

    def test_build_step(self):
        step = _build_step_from_particles(
            u=jnp.array([[0.0], [1.0]]),
            x=jnp.array([[2.0], [3.0]]),
            logdetj=jnp.array([0.1, 0.2]),
            logl=jnp.array([1.0, 2.0]),
            logp=jnp.array([0.5, 0.6]),
            blobs=jnp.zeros((2, 0)),
            iter_idx=jnp.array(3),
            beta=jnp.array(0.7),
            logz=jnp.array(1.2),
            calls=jnp.array(5.0),
            steps=jnp.array(2.0),
            efficiency=jnp.array(0.8),
            ess=jnp.array(1.5),
            accept=jnp.array(0.4),
        )
        assert step.x.shape == (2, 1)
        assert step.logw.shape == (2,)
        assert step.iter == 3

    def test_sampler_run(self):
        prior = Prior.create([NORMAL], [[0.0, 1.0]])
        cfg = SamplerConfigJAX(
            n_dim=1,
            n_active=8,
            n_effective=4,
            n_prior=16,
            n_total=5,
            n_steps=1,
            n_max_steps=2,
            keep_max=8,
            preconditioned=False,
        )
        sampler = SamplerJAX(
            prior,
            lambda x: jnp.array(-0.5 * jnp.sum((x - 0.2) ** 2), dtype=jnp.float64),
            cfg,
        )
        out = sampler.run(jax.random.key(0))
        assert out.state.t > 0
        assert jnp.isfinite(out.logz)


if __name__ == "__main__":
    absltest.main()
