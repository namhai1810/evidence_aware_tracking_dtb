from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from hive_ecia.dataset import SAMPLE_KIND_TO_ID
from hive_ecia.model import ECIA
from hive_ecia.recovery_aware import RecoveryAwareECIA, total_parameter_count, trainable_parameter_count
from hive_ecia.recovery_aware_loss import RecoveryAwareLoss


class RecoveryAwareECIATest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.inputs = {
            "old": torch.randn(4, 512),
            "target": torch.randn(4, 512),
            "hr": torch.randn(4, 512),
            "competitors": torch.randn(4, 3, 512),
            "competitor_mask": torch.tensor(
                [[True, True, False], [True, True, True], [True, False, False], [False, False, False]]
            ),
            "hr_available": torch.tensor([True, True, True, False]),
        }

    def test_parameter_budget_and_frozen_base(self) -> None:
        model = RecoveryAwareECIA()
        self.assertLess(total_parameter_count(model), 50_000)
        self.assertEqual(trainable_parameter_count(model), 6_978)
        for name, parameter in model.named_parameters():
            if name.startswith(model.FROZEN_PREFIXES):
                self.assertFalse(parameter.requires_grad, name)
            else:
                self.assertTrue(parameter.requires_grad, name)

    def test_gate_zero_exactly_reproduces_transferred_base(self) -> None:
        base = ECIA()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "base.pt"
            torch.save({"ablation": "base_only", "model": base.state_dict()}, path)
            model = RecoveryAwareECIA()
            model.load_frozen_base(path)
        base.eval()
        model.eval()
        with torch.no_grad():
            expected = base(**self.inputs, ablation="base_only")
            actual = model(**self.inputs, gate_override=0.0)
        self.assertTrue(torch.allclose(actual.base_scores, expected.base_scores, atol=1e-7))
        self.assertTrue(torch.allclose(actual.abstain_logit, expected.abstain_logit, atol=1e-7))
        self.assertTrue(torch.allclose(actual.decision_logits, expected.decision_logits, atol=1e-7))

    def test_recovery_loss_updates_only_hr_modules(self) -> None:
        model = RecoveryAwareECIA()
        output = model(**self.inputs)
        criterion = RecoveryAwareLoss()
        losses = criterion(
            output,
            target_class=torch.tensor([0, 1, 0, 4]),
            merge_target=torch.tensor([1.0, 0.0, 1.0, 0.0]),
            sample_kind=torch.tensor(
                [
                    SAMPLE_KIND_TO_ID["positive"],
                    SAMPLE_KIND_TO_ID["negative"],
                    SAMPLE_KIND_TO_ID["hard_negative"],
                    SAMPLE_KIND_TO_ID["no_merge"],
                ]
            ),
        )
        losses.total.backward()
        self.assertTrue(torch.isfinite(losses.total))
        for name, parameter in model.named_parameters():
            if name.startswith(model.FROZEN_PREFIXES):
                self.assertIsNone(parameter.grad, name)
            else:
                self.assertIsNotNone(parameter.grad, name)

    def test_missing_hr_forces_zero_gate(self) -> None:
        model = RecoveryAwareECIA()
        inputs = dict(self.inputs)
        inputs["hr_available"] = torch.zeros(4, dtype=torch.bool)
        first = model(**inputs)
        inputs["hr"] = torch.randn_like(inputs["hr"]) * 100
        second = model(**inputs)
        self.assertTrue(torch.equal(first.gate, torch.zeros_like(first.gate)))
        self.assertTrue(torch.allclose(first.decision_logits, second.decision_logits, atol=1e-7))


if __name__ == "__main__":
    unittest.main()
