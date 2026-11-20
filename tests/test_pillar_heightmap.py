'''核查点柱的坐标、未来回波隔离、分层输出与检查点。'''

import io
import os
import unittest

import torch
from torch.nn import functional as F

from rsl_rl.models.pillar_heightmap import PillarHeightMap


DEVICE = os.environ.get("PILLAR_SMOKE_DEVICE", "cpu")


class PillarHeightMapTest(unittest.TestCase):
    '''用可手算的同柱回波和共用网格检查三个变体。'''

    def test_column_coordinates_and_layers(self):
        '''
        同一格放入踏面和上方回波，验证单值与点柱的可区分性。
        测试空格、未来／无效点、帧隔离和点顺序。
        '''
        xyz = torch.tensor(
            [[[
                [0.025, 0.025, -1.0],
                [0.024, 0.026, -0.4],
                [0.025, 0.025, 10.0],
                [float("nan"), 0.025, 10.0],
                [0.025, 0.025, -10.0],
                [0.075, 0.125, -0.9],
            ], [
                [0.025, 0.025, -0.6],
                [0.25, 0.025, 9.0],
                [0.025, -0.02, 9.0],
                [0.025, 0.025, -9.0],
                [float("inf"), 0.025, 9.0],
                [0.025, 0.025, 9.0],
            ]]],
            device=DEVICE,
        )
        age_s = torch.tensor([[[0.11, 0.11, -0.01, 0.11, 0.09, 0.09], [0.0] * 6]], device=DEVICE)
        valid = torch.tensor([[[True, True, True, True, False, True], [True, True, True, False, True, False]]], device=DEVICE)
        models = {
            name: PillarHeightMap(scans=2, shape_xy=(5, 4), lower_xy_m=(0.0, 0.0), representation=name, width=4)
            .to(DEVICE)
            for name in ("single", "statistics", "pillars")
        }
        with torch.no_grad():
            descriptor = models["pillars"].point_encoder[0]
            descriptor.weight.zero_()
            descriptor.bias.fill_(1.0)
            descriptor.weight[0, 2] = 1.0  # 明确的高度探针，池化后应保留上方回波。
        for name, model in models.items():
            with self.subTest(representation=name):
                grid = model.rasterize(xyz, age_s, valid)
                torch.testing.assert_close(grid, model.rasterize(xyz.flip(2), age_s.flip(2), valid.flip(2)))
                self.assertAlmostEqual(grid[0, 0, 0, 0, 0].item(), -1.0)
                self.assertAlmostEqual(grid[0, 0, 0, 1, 2].item(), -0.9)
                self.assertAlmostEqual(grid[0, 1, 0, 0, 0].item(), -0.6)
                self.assertEqual(grid[0, 0, :, 2, 0].abs().sum().item(), 0.0)
                self.assertEqual(grid[0, 1, 2].sum().item(), 1.0)

        without_upper = valid.clone()
        without_upper[..., 0, 1] = False
        single = models["single"]
        torch.testing.assert_close(single.rasterize(xyz, age_s, without_upper), single.rasterize(xyz, age_s, valid))
        before = models["pillars"].rasterize(xyz, age_s, without_upper)
        after = models["pillars"].rasterize(xyz, age_s, valid)
        self.assertAlmostEqual(before[0, 0, 0, 0, 0].item(), after[0, 0, 0, 0, 0].item())
        self.assertAlmostEqual(before[0, 0, 3, 0, 0].item(), 0.0)
        self.assertAlmostEqual(after[0, 0, 3, 0, 0].item(), 0.6)
        stats = models["statistics"].rasterize(xyz, age_s, valid)
        torch.testing.assert_close(stats[0, 0, 3:7, 0, 0], xyz.new_tensor([-1.0, -0.7, -0.4, 0.3]))

    def test_update_and_restore(self):
        '''
        使用 5 扫描和 28×20 网格检查所有变体的反向传播。
        将分层版保存到内存再载入，验证恢复后的输出。
        '''
        torch.manual_seed(77)
        batch, scans, rays = 2, 5, 20_000  # 共用考卷单次完整扫描的射线数。
        xyz = 4 * torch.rand(batch, scans, rays, 3, device=DEVICE) - 2.0  # 包含落在足端网格外的回波。
        age_s = torch.linspace(0.4, 0.0, scans, device=DEVICE)[None, :, None].expand(batch, scans, rays)
        valid = torch.rand(batch, scans, rays, device=DEVICE) > 0.2
        grid = dict(scans=scans, shape_xy=(28, 20), lower_xy_m=(-0.7, -0.5))
        configs = (("single", 0), ("statistics", 0), ("pillars", 0), ("pillars", 12))
        for representation, layers in configs:
            with self.subTest(representation=representation, layers=layers):
                setup = dict(**grid, representation=representation, layers=layers)
                model = PillarHeightMap(**setup).to(DEVICE)
                outputs = model(xyz, age_s, valid)
                self.assertEqual(outputs["height_m"].shape, (batch, 28, 20))
                loss = F.l1_loss(outputs["height_m"], torch.zeros_like(outputs["height_m"]))
                loss = loss + outputs["support_logits"].square().mean() + outputs["coverage_logits"].square().mean()
                if layers:
                    occupancy = outputs["occupancy_logits"]
                    self.assertEqual(occupancy.shape, (batch, layers, 28, 20))
                    loss = loss + F.binary_cross_entropy_with_logits(occupancy, torch.zeros_like(occupancy))
                loss.backward()
                for parameter in model.parameters():
                    self.assertTrue(parameter.grad.isfinite().all().item())
                torch.optim.AdamW(model.parameters(), lr=1e-3).step()
                if layers:
                    stream = io.BytesIO()
                    torch.save(model.state_dict(), stream)
                    stream.seek(0)
                    restored = PillarHeightMap(**setup).to(DEVICE)
                    restored.load_state_dict(torch.load(stream, map_location=DEVICE, weights_only=True))
                    with torch.no_grad():
                        expected = model(xyz, age_s, valid)
                        torch.testing.assert_close(restored(xyz, age_s, valid), expected)


if __name__ == "__main__":
    unittest.main()
