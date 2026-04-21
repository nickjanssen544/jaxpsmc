import chex
import jax.numpy as jnp
from absl.testing import absltest

from jaxpsmc.input_validation_jax import (
    assert_array_1d,
    assert_array_2d,
    assert_array_float,
    assert_array_within_interval,
    assert_arrays_equal_shape,
    assert_equal_type,
    jit_with_checks,
    within_interval_mask,
)


class InputCheckTest(chex.TestCase):
    def test_ndim_ok(self):
        checked = jit_with_checks(assert_array_2d)
        x = jnp.ones((2, 3), dtype=jnp.float32)
        y = checked(x)
        assert y.shape == (2, 3)

    def test_ndim_fail(self):
        checked = jit_with_checks(assert_array_1d)
        with self.assertRaisesRegex(Exception, "dimensions"):
            checked(jnp.ones((2, 3)))

    def test_shape_and_dtype(self):
        checked = jit_with_checks(lambda x, y: assert_equal_type(*assert_arrays_equal_shape(x, y)))
        x = jnp.ones((2, 2), dtype=jnp.float32)
        y = jnp.zeros((2, 2), dtype=jnp.float32)
        x2, y2 = checked(x, y)
        assert x2.shape == y2.shape

    def test_float_fail(self):
        checked = jit_with_checks(assert_array_float)
        with self.assertRaisesRegex(Exception, "floating dtype"):
            checked(jnp.array([1, 2, 3], dtype=jnp.int32))

    @chex.all_variants(with_pmap=False)
    def test_interval_mask(self):
        x = jnp.array([0.0, 1.0, 2.0])
        mask = self.variant(lambda z: within_interval_mask(z, 0.0, 2.0, left_open=True))(x)
        assert mask.tolist() == [False, True, True]

    def test_interval_fail(self):
        checked = jit_with_checks(assert_array_within_interval)
        with self.assertRaisesRegex(Exception, "outside the required interval"):
            checked(jnp.array([0.0, 3.0]), 0.0, 2.0)


if __name__ == "__main__":
    absltest.main()
