"""Numerical checks of experiment contracts without simulator/GPU requirements."""

import unittest

import torch

from rsl_rl.models.range_experiments import RangeExperiment
from rsl_rl.models.range_jepa import RangeJEPA


def image(batch=2, height=12, width=32):
    depth = torch.rand(batch, 1, height, width) * 0.15 + 0.025
    mask = (torch.rand_like(depth) > 0.2).float()
    return torch.cat((depth * mask, mask), 1)


class RangeExperimentTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)

    def model(self, variant="C", **kwargs):
        return RangeExperiment(image_height=12, image_width=32, variant=variant, **kwargs)

    def history(self, model, batch=2, frames=3):
        encoded = model.encode_history(image(batch * frames))
        return dict(
            history_tokens=encoded["tokens"].reshape(batch, frames, model.history_token_count, 32),
            history_valid=encoded["valid"].reshape(batch, frames, model.history_token_count),
            history_age=torch.arange(1, frames + 1)[None].expand(batch, -1) * 0.1,
        )

    def test_shapes_and_live_history_encoder_reuse(self):
        for variant in ("A1", "B", "C", "D", "E", "F"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                x = image()
                out = model.step(x, model.initial_hidden(2))
                self.assertEqual(out["features"].shape, (2, model.feature_size))
                self.assertTrue(torch.isfinite(out["features"]).all())
                if variant != "A1":
                    expected = model.encode_history(x)
                    torch.testing.assert_close(out["history_encoded"]["tokens"], expected["tokens"])
                    torch.testing.assert_close(out["history_encoded"]["valid"], expected["valid"])
                if variant not in ("A1", "D"):
                    self.assertFalse(hasattr(model, "gru"))
                expected_shape = (2, 2, 2, 32, 32) if variant == "F" else (2, 2, 12, 32)
                self.assertEqual(model.decode(out["z"]).shape, expected_shape)

    def test_a1_predictor_identical_to_a0(self):
        base = RangeJEPA(image_height=12, image_width=32)
        model = self.model("A1")
        state = {k: v for k, v in base.state_dict().items() if not k.startswith("decoder.")}
        model.load_state_dict(state, strict=False)
        x, hidden = image(), torch.randn(2, 128)
        old, new = base.step(x, hidden), model.step(x, hidden)
        for key in ("e", "z", "features", "hidden"):
            torch.testing.assert_close(old[key], new[key])

    def test_history_and_current_content_affect_prediction(self):
        for variant in ("B", "C", "D", "F"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                x, h = image(), model.initial_hidden(2)
                history = self.history(model)
                out = model.step(x, h, **history)
                empty = model.step(x, h)
                changed = model.step(x.roll(5, -1), h, **history)
                self.assertGreater((out["e"] - empty["e"]).abs().max().item(), 1e-5)
                self.assertGreater((out["e"] - changed["e"]).abs().max().item(), 1e-5)

    def test_invalid_or_reset_history_has_no_influence(self):
        for variant in ("B", "C", "D", "F"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                x, h = image(), torch.randn(2, 128)
                history = self.history(model)
                reset = torch.ones(2, dtype=torch.bool)
                actual = model.step(x, h, reset=reset, **history)
                expected = model.step(x, model.initial_hidden(2))
                torch.testing.assert_close(actual["features"], expected["features"], atol=2e-6, rtol=2e-5)
                history["history_valid"] = torch.zeros_like(history["history_valid"])
                history["history_tokens"] = torch.full_like(history["history_tokens"], float("nan"))
                history["history_age"] = torch.full_like(history["history_age"], float("nan"))
                actual = model.step(x, h, **history)
                expected = model.step(x, h)
                torch.testing.assert_close(actual["features"], expected["features"], atol=2e-6, rtol=2e-5)

    def test_empty_current_and_history_remain_finite(self):
        model = self.model("C")
        empty = torch.zeros(2, 2, 12, 32)
        out = model.step(empty, model.initial_hidden(2))
        self.assertTrue(torch.isfinite(out["features"]).all())
        loss = model.auxiliary_loss(empty, empty, empty)
        self.assertTrue(torch.isfinite(loss["loss"]))
        loss["loss"].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_history_offsets_match_named_windows(self):
        self.assertEqual(self.model("B").history_offsets, tuple(range(1, 20)))
        self.assertEqual(self.model("C").history_offsets, tuple(range(1, 10)) + (10, 20, 30, 40, 50))
        self.assertEqual(
            self.model("C", history_seconds=10).history_offsets, tuple(range(1, 10)) + tuple(range(10, 101, 10))
        )
        self.assertEqual(self.model("E").history_offsets, self.model("F").history_offsets)

    def test_meter_loss_is_not_normalized_pixel_loss(self):
        model = self.model("A1", edge_weight=0, near_weight=0)
        target = torch.ones(2, 2, 12, 32)
        target[:, 0] = 2 / 40  # 2 metres
        prediction = target.clone()
        prediction[:, 0] = 3 / 40  # 1 metre error
        loss = model._reconstruction(prediction, target, predict_mask=False)
        torch.testing.assert_close(loss, torch.full_like(loss, 0.1 * (1 - 0.05)))
        target[:, 1] = 0
        torch.testing.assert_close(model._reconstruction(prediction, target, predict_mask=False), torch.zeros(2))

    def test_loss_backpropagates_encoder_predictor_decoder(self):
        for variant in ("A1", "B", "C", "D", "E", "F"):
            with self.subTest(variant=variant):
                model = self.model(variant)
                obs, mapped, sim = image(), image(), image()
                context = {} if variant == "A1" else self.history(model)
                loss = model.auxiliary_loss(obs, mapped, sim, **context)
                self.assertTrue(torch.isfinite(loss["loss"]))
                loss["loss"].backward()
                self.assertGreater(model.encoder[0].conv.weight.grad.abs().sum().item(), 0)
                self.assertGreater(model.readout.in_proj_weight.grad.abs().sum().item(), 0)
                if variant == "F":
                    self.assertGreater(model.bev_backend.target_encoder[0].weight.grad.abs().sum().item(), 0)
                    self.assertGreater(model.bev_backend.decoder[-1].weight.grad.abs().sum().item(), 0)
                else:
                    self.assertGreater(model.decoder.head.weight.grad.abs().sum().item(), 0)
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                no_grad = [name for name, param in model.named_parameters() if param.grad is None]
                self.assertEqual(no_grad, [], f"Dormant parameters in {variant}: {no_grad}")

    def test_raw_history_optional_gradients_and_active_padding(self):
        model = self.model("C")
        observed = torch.stack((image(), image()))
        history_images = torch.stack(
            (image(6).reshape(2, 3, 2, 12, 32), image(6).reshape(2, 3, 2, 12, 32))
        ).requires_grad_()
        age = torch.full((2, 2, 3), 0.2)
        active = torch.tensor([[True, True], [False, False]])
        loss = model.auxiliary_loss(
            observed, observed, observed, history_images=history_images, history_age=age, active=active
        )
        loss["loss"].backward()
        self.assertGreater(history_images.grad[0].abs().sum().item(), 0)
        torch.testing.assert_close(history_images.grad[1], torch.zeros_like(history_images.grad[1]))
        self.assertEqual(loss["anchor_count"].item(), 2)


if __name__ == "__main__":
    unittest.main()
