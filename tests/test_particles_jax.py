import chex
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.particles_jax import (
    ParticlesStep,
    compute_logw_and_logz_jax,
    compute_results_jax,
    init_particles_state_jax,
    pop_step_jax,
    record_step_jax,
    step_mask_jax,
)


class ParticlesTest(chex.TestCase):
    def _step(self, shift, beta=0.0, logz=0.0):
        u = jnp.array([[shift], [shift + 1.0]])
        x = u + 10.0
        logl = jnp.array([shift, shift + 1.0])
        logp = -u[:, 0] * 0.5
        z = jnp.zeros((2,))
        return ParticlesStep(
            u=u,
            x=x,
            logdetj=z,
            logl=logl,
            logp=logp,
            logw=z,
            blobs=jnp.zeros((2, 0)),
            iter=jnp.array(shift, dtype=jnp.int32),
            logz=jnp.array(logz),
            calls=jnp.array(2.0),
            steps=jnp.array(1.0),
            efficiency=jnp.array(1.0),
            ess=jnp.array(2.0),
            accept=jnp.array(1.0),
            beta=jnp.array(beta),
        )

    def test_init(self):
        state = init_particles_state_jax(4, 2, 1)
        assert state.t == 0
        assert state.u.shape == (4, 2, 1)
        assert jnp.all(jnp.isneginf(state.logw))

    def test_record_and_pop(self):
        state = init_particles_state_jax(3, 2, 1)
        state = record_step_jax(state, self._step(0.0))
        state = record_step_jax(state, self._step(1.0))
        assert state.t == 2
        np.testing.assert_array_equal(step_mask_jax(state), jnp.array([True, True, False]))
        state = pop_step_jax(state)
        assert state.t == 1

    @chex.all_variants(with_pmap=False)
    def test_logw_single_step(self):
        state = init_particles_state_jax(3, 2, 1)
        state = record_step_jax(state, self._step(0.0, beta=0.0, logz=0.0))
        logw, logz, mask = self.variant(compute_logw_and_logz_jax)(state, beta_final=1.0)
        expected = jnp.array([0.0, 1.0])
        expected = expected - jax.nn.logsumexp(expected)
        np.testing.assert_allclose(logw[:2], expected, atol=1e-6)
        np.testing.assert_array_equal(mask, jnp.array([True, True, False, False, False, False]))
        np.testing.assert_allclose(logz, jax.nn.logsumexp(jnp.array([0.0, 1.0])) - jnp.log(2.0))

    @chex.all_variants(with_pmap=False)
    def test_results(self):
        state = init_particles_state_jax(2, 2, 1)
        state = record_step_jax(state, self._step(0.0))
        out = self.variant(compute_results_jax)(state)
        assert out["t"] == 1
        assert out["mask_t"].tolist() == [True, False]
        assert out["x"].shape == (2, 2, 1)


if __name__ == "__main__":
    absltest.main()
