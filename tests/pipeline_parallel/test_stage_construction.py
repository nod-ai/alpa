import unittest

from alpa.pipeline_parallel.stage_construction import AutoStageOption
from alpa.testing import PipelineBasicTest
from alpa.parallel_method import PipeshardParallel, InterOpConfig
from alpa import parallelize, get_global_cluster, InvalidExecutable
import jax.numpy as jnp


def auto_stage():
    return AutoStageOption("small_power_of_two", "default", float("inf"), False,
                           None, None)


class StageConstructionTest(PipelineBasicTest):

    def test_mlp_stage_construction(self):
        self.run_mlp(stage_option=auto_stage())

    def test_mlp_layer_and_stage(self):
        self.run_mlp(manual_pipeline_layer=False, stage_option=auto_stage())

    def test_2_layer_bert_stage_construction(self):
        self.run_n_layer_bert(n_layers=2, stage_option=auto_stage())

    def test_2_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(n_layers=2,
                              manual_pipeline_layer=False,
                              stage_option=auto_stage())

    def test_8_layer_bert_stage_construction(self):
        self.run_n_layer_bert(n_layers=8, stage_option=auto_stage())

    def test_8_layer_bert_layer_and_stage(self):
        self.run_n_layer_bert(n_layers=8,
                              manual_pipeline_layer=False,
                              stage_option=auto_stage())

    def test_disabled_inter_op_construction(self):
        with self.assertRaises(RuntimeError,
                               msg="Inter-op construction is disabled."):
            self.run_mlp(method=PipeshardParallel(
                num_micro_batches=1,
                stage_mode="auto",
                inter_op_config=InterOpConfig(
                    is_inter_op_construction_enabled=False),
            ))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(StageConstructionTest('test_mlp_stage_construction'))
    suite.addTest(StageConstructionTest('test_mlp_layer_and_stage'))
    suite.addTest(StageConstructionTest('test_2_layer_bert_stage_construction'))
    suite.addTest(StageConstructionTest('test_2_layer_bert_layer_and_stage'))
    suite.addTest(StageConstructionTest('test_8_layer_bert_stage_construction'))
    suite.addTest(StageConstructionTest('test_8_layer_bert_layer_and_stage'))
    suite.addTest(StageConstructionTest('test_disabled_inter_op_construction'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
