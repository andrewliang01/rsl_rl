from contextlib import redirect_stdout
import io
import unittest
import torch
from rsl_rl.algorithms.range_jepa_amp_ppo import RangeJEPAAMPPPO, _model_configuration
from test_range_jepa_amp_ppo import TinyEnv, cfg


class RangeCudnnConfigurationTests(unittest.TestCase):
    def setUp(self):self.original=torch.backends.cudnn.benchmark
    def tearDown(self):torch.backends.cudnn.benchmark=self.original

    def construct(self, value='omitted'):
        env=TinyEnv();configuration=cfg()
        if value!='omitted':configuration['algorithm']['perception_cfg']['cudnn_benchmark']=value
        return RangeJEPAAMPPPO.construct_algorithm(env.get_observations(),env,configuration,'cpu')

    def test_explicit_false_overrides_env_seed_true_and_is_not_model_configuration(self):
        torch.backends.cudnn.benchmark=True
        output=io.StringIO()
        with redirect_stdout(output):alg=self.construct(False)
        self.assertFalse(torch.backends.cudnn.benchmark)
        self.assertNotIn('cudnn_benchmark',alg.perception_model_cfg)
        self.assertIn('requested=False, previous=True, actual=False',output.getvalue())
        self.assertFalse(alg.last_perception_receipt['cudnn_benchmark_actual'])
        self.assertFalse(alg.last_perception_receipt['cudnn_benchmark_requested'])
        self.assertNotIn('cudnn_benchmark',_model_configuration({'image_height':8,'image_width':16,'cudnn_benchmark':False}))

    def test_default_and_none_preserve_existing_global_value(self):
        for existing in (True,False):
            for value in ('omitted',None):
                torch.backends.cudnn.benchmark=existing
                output=io.StringIO()
                with redirect_stdout(output):alg=self.construct(value)
                self.assertEqual(torch.backends.cudnn.benchmark,existing)
                self.assertIsNone(alg.cudnn_benchmark_requested)
                self.assertNotIn('RangeJEPA cuDNN benchmark:',output.getvalue())

    def test_true_and_invalid_types(self):
        torch.backends.cudnn.benchmark=False
        with redirect_stdout(io.StringIO()):alg=self.construct(True)
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertTrue(alg.cudnn_benchmark_requested)
        for invalid in (0,1,'false',[],{}):
            torch.backends.cudnn.benchmark=True
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError,'must be bool or None'):self.construct(invalid)
            self.assertTrue(torch.backends.cudnn.benchmark)


if __name__=='__main__':unittest.main(verbosity=2)
