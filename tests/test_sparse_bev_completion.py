import io
import unittest

import torch

from rsl_rl.models.sparse_bev_completion import SparseBEVCompleter


class SparseBEVCompletionTest(unittest.TestCase):

    def test_invalid_values_and_exact_measurements(self):
        '''
        检查奇数网格、NaN 哨兵和完全空图。
        输出形状须一致，可靠格逐值保留，空格保持稠密有限且不计为实测。
        '''
        model = SparseBEVCompleter(channels=4)
        height = torch.full((2, 1, 19, 27), float("nan"))
        observed = torch.zeros_like(height, dtype=torch.bool)
        height[0, 0, 2, 3] = -0.63
        observed[0, 0, 2, 3] = True
        observed[0, 0, 4, 5] = True  # 损坏的回波仍按缺测处理。
        result = model(height, observed)
        self.assertEqual(result["height"].shape, (2, 19, 27))
        self.assertEqual(result["height"][0, 2, 3].item(), height[0, 0, 2, 3].item())
        self.assertEqual(result["observed"].sum().item(), 1)
        self.assertTrue(result["height"].isfinite().all())
        self.assertTrue(torch.equal(result["height"], model(height.nan_to_num(), height.isfinite())["height"]))

    def test_shared_single_and_multiframe_contract(self):
        '''
        检查历史显式空帧与单帧输出一致，且过去命中不改变当前观测掩码。
        使用 [B,T,X,Y] 的 5 cm 足端网格形状，输出为同尺寸高程。
        '''
        model = SparseBEVCompleter(channels=4)
        height = torch.randn(1, 5, 28, 20)
        observed = torch.zeros_like(height, dtype=torch.bool)
        age = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0]])
        with self.assertRaises(ValueError):
            model(height, observed)
        full = model(height, observed, age)
        single = model(height[:, -1:], observed[:, -1:])
        torch.testing.assert_close(full["height"], single["height"])
        observed[:, 0, 2, 3] = True
        self.assertFalse(model(height, observed, age)["observed"].any())

    def test_learns_hidden_grid_and_reloads_exactly(self):
        '''
        检查缺测格梯度可更新，序列化并重载后推理与保存前一致。
        输入使用当前部分命中的足端图，输出用于模拟补洞损失。
        '''
        model = SparseBEVCompleter(channels=4)
        height = torch.ones(2, 1, 28, 20)
        observed = torch.zeros_like(height, dtype=torch.bool)
        observed[:, :, :, ::3] = True
        result = model(height, observed)
        result["height"][~result["observed"]].square().mean().backward()
        self.assertGreater(model.decode_full[-1].weight.grad.abs().sum().item(), 0)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        restored = SparseBEVCompleter(channels=4)
        restored.load_state_dict(torch.load(buf, weights_only=True))
        torch.testing.assert_close(restored(height, observed)["height"], result["height"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
