import unittest
import os

import jax
import jax.numpy as jnp
import optax
import ray

from alpa import (init, parallelize, automatic_layer_construction,
                  PipeshardParallel)
from alpa.model.model_util import TrainState
from alpa.testing import MLPModel, assert_allclose
from alpa.shard_parallel.auto_sharding import AutoShardingOption


class NodAutoShardingTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        init(cluster="ray")

    def train_2_layer_mlp(self, method):

        def train_step(state, batch):

            @automatic_layer_construction(layer_num=2)
            def loss_func(params, x, y):
                out = state.apply_fn(params, x)
                loss = jnp.mean((out - y)**2)
                return loss

            grads = jax.grad(loss_func)(state.params, batch["x"], batch["y"])
            return grads

        batch_size = 64
        hidden_dim = 1024
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = MLPModel(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.sgd(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Train step
        batch = {"x": x, "y": y}
        gradients = train_step(state, batch)
        p_train_step = parallelize(train_step, donate_argnums=(), method=method)
        gradients_with_pipeline = p_train_step(state, batch)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)

        # Check debug utilities
        if isinstance(method, PipeshardParallel):
            executable = p_train_step.get_last_executable()
            executable.dump_debug_info("tmp")

    def test_smoke(self):
        auto_sharding_opts = AutoShardingOption(algorithm="nod")
        self.train_2_layer_mlp(
            PipeshardParallel(default_auto_sharding_option=auto_sharding_opts,
                              stage_mode="auto"))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(NodAutoShardingTest("test_smoke"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
