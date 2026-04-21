import chex
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from jaxpsmc.input_validation_jax import jit_with_checks
from jaxpsmc.scaler_jax import (
    apply_boundary_conditions_x_jax,
    apply_periodic_boundary_conditions_x_jax,
    apply_reflective_boundary_conditions_x_jax,
    fit_jax,
    forward_jax,
    forward_jax_checked,
    init_bounds_config_jax,
    inverse_jax,
    masks_jax,
)


class ScalerCfgTest(chex.TestCase):
    def test_init_config(self):
        cfg = init_bounds_config_jax(3, bounds=jnp.array([[0.0, 1.0], [0.0, 2.0], [-jnp.inf, jnp.inf]]))
        assert cfg["low"].shape == (3,)
        assert cfg["high"].shape == (3,)
        assert cfg["transform_id"] == 1



class ScalerTransformTest(chex.TestCase):
    @chex.all_variants(with_pmap=False)
    def test_roundtrip_unscaled(self):
        cfg = init_bounds_config_jax(2, bounds=jnp.array([[0.0, 1.0], [-jnp.inf, jnp.inf]]), scale=False)
        masks = masks_jax(cfg["low"], cfg["high"])
        x = jnp.array([[0.2, -1.0], [0.8, 2.0]])
        u = self.variant(lambda z: forward_jax(z, cfg, masks))(x)
        x2, ld = self.variant(lambda z: inverse_jax(z, cfg, masks))(u)
        np.testing.assert_allclose(x2, x, atol=1e-6)
        assert ld.shape == (2,)

    @chex.all_variants(with_pmap=False)
    def test_fit_and_roundtrip_scaled(self):
        x = jnp.array([[0.1], [0.3], [0.7], [0.9]])
        cfg = init_bounds_config_jax(1, bounds=jnp.array([0.0, 1.0]), scale=True, diagonal=True)
        masks = masks_jax(cfg["low"], cfg["high"])
        fit_cfg = self.variant(lambda z: fit_jax(z, cfg, masks))(x)
        u = forward_jax(x, fit_cfg, masks)
        x2, _ = inverse_jax(u, fit_cfg, masks)
        np.testing.assert_allclose(x2, x, atol=1e-6)
        assert jnp.isfinite(fit_cfg["mu"]).all()
        assert jnp.isfinite(fit_cfg["sigma"]).all()

    def test_forward_checked_fail(self):
        checked = jit_with_checks(lambda x, cfg, masks: forward_jax_checked(x, cfg, masks))
        cfg = init_bounds_config_jax(1, bounds=jnp.array([0.0, 1.0]), scale=False)
        masks = masks_jax(cfg["low"], cfg["high"])
        with self.assertRaisesRegex(Exception, "outside the required interval"):
            checked(jnp.array([[1.5]]), cfg, masks)

    def test_boundary_conditions(self):
        low = jnp.array([0.0, 0.0])
        high = jnp.array([2.0, 2.0])
        x = jnp.array([[2.5, -0.5], [4.0, 5.0]])
        x_ref = apply_reflective_boundary_conditions_x_jax(x, low, high, jnp.array([True, False]))
        x_per = apply_periodic_boundary_conditions_x_jax(x, low, high, jnp.array([False, True]))
        cfg = {
            "low": low,
            "high": high,
            "periodic_mask": jnp.array([False, True]),
            "reflective_mask": jnp.array([True, False]),
        }
        x_all = apply_boundary_conditions_x_jax(x, cfg)
        assert jnp.all((x_ref[:, 0] >= 0.0) & (x_ref[:, 0] <= 2.0))
        assert jnp.all((x_per[:, 1] >= 0.0) & (x_per[:, 1] <= 2.0))
        np.testing.assert_allclose(x_all[:, 0], x_ref[:, 0])
        np.testing.assert_allclose(x_all[:, 1], x_per[:, 1])


if __name__ == "__main__":
    absltest.main()
