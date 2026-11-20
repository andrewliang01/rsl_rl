"""原始点集模型的形状、因果遮罩及训练检查。"""

import io
import unittest

import torch

from rsl_rl.models.raw_point_heightmap import RawPointHeightmap


class TestRawPointHeightmap(unittest.TestCase):
    def test_causal_masks_and_point_order(self):
        '''
        输入有缺测、未来观测及打乱点序的雷达窗口。
        检查屏蔽等价、点序不影响预测及正确展开的固定格心。
        '''
        generator = torch.Generator().manual_seed(25)
        model = RawPointHeightmap().eval()
        points = torch.randn(2, 5, 23, 3, generator=generator) * 0.2
        valid = torch.ones(2, 5, 23, dtype=torch.bool)
        age = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0]]).expand(2, -1).clone()
        future_age = age.clone()
        future_age[:, :4] *= -1
        valid[:, -1, 0] = False
        points[:, -1, 0] = torch.nan

        with torch.no_grad():
            history = model(points, valid, future_age)
            single = model(points[:, -1:], valid[:, -1:], age[:, -1:])
            torch.testing.assert_close(history, single)
            order = torch.randperm(points.size(2), generator=generator)
            per_point_age = future_age.unsqueeze(-1).expand_as(valid)
            permuted = model(points[:, :, order], valid[:, :, order], per_point_age[:, :, order])
            torch.testing.assert_close(permuted, single)
            self.assertFalse(torch.allclose(model(points, valid, age).height_m, single.height_m))
            empty = model(points, ~torch.ones_like(valid), age)

        self.assertEqual(history.height_m.shape, (2, 560))
        self.assertTrue(torch.isfinite(empty.height_m).all())
        self.assertTrue(torch.isfinite(empty.observed_logits).all())
        torch.testing.assert_close(model.query_xy_m[[0, 1, 20, -1]], torch.tensor(
            [[-0.675, -0.475], [-0.675, -0.425], [-0.625, -0.475], [0.675, 0.475]]
        ))

    def test_five_scan_update_and_checkpoint(self):
        '''
        用不同帧龄和三处查询检查反传、权重更新及保存回载。
        输入为五扫小批和自定义格心，输出形状及回载预测应一致。
        '''
        generator = torch.Generator().manual_seed(75)
        model = RawPointHeightmap(neighbors=8)
        points = torch.randn(2, 5, 12, 3, generator=generator) * 0.1
        valid = torch.ones(2, 5, 12, dtype=torch.bool)
        age = torch.tensor([[0.4, 0.3, 0.2, 0.1, 0.0]]).expand(2, -1)
        query = torch.tensor([[0., 0.], [0.12, 0.03], [-0.21, 0.10]])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        result = model(points, valid, age, query)
        self.assertEqual(result.height_m.shape, (2, 3))
        loss = (result.height_m - 0.7).square().mean() + result.observed_logits.square().mean()
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()

        with io.BytesIO() as checkpoint:
            torch.save(model.state_dict(), checkpoint)
            checkpoint.seek(0)
            restored = RawPointHeightmap(neighbors=8)
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        expected = model(points, valid, age, query)
        self.assertFalse(torch.equal(result.height_m, expected.height_m))
        torch.testing.assert_close(restored(points, valid, age, query), expected)
