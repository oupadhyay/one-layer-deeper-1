import unittest

import torch
import torch.nn.functional as F

from benchmark import ModelSpec, OptimizerSpec, count_model_state_elements
import competition_submission as c1
import competition_submission_c1a as c1a
import competition_submission_c1b as c1b


SPEC = ModelSpec(17, 13, 500_000_000)


def prompts():
    return torch.tensor([
        [c1.N, 10, 9, 10, c1.X, 8, 12, c1.T, 8],
        [c1.N, 10, 9, 10, c1.X, 15, 16, c1.T, 9],
        [c1.N, 10, 9, 10, c1.X, 11, 9, c1.T, 10],
    ])


class C1BTests(unittest.TestCase):
    def paired_models(self):
        torch.manual_seed(74)
        parent = c1.build_model(SPEC)
        torch.manual_seed(74)
        ablation = c1b.build_model(SPEC)
        return parent, ablation

    def test_initialization_state_and_optimizer_are_identical(self):
        parent, ablation = self.paired_models()
        self.assertEqual(count_model_state_elements(parent), count_model_state_elements(ablation))
        self.assertEqual(parent.state_dict().keys(), ablation.state_dict().keys())
        for key in parent.state_dict():
            self.assertTrue(torch.equal(parent.state_dict()[key], ablation.state_dict()[key]))
        parent_bundle = c1a.build_optimizer(parent, OptimizerSpec(60, "cpu"))
        ablation_bundle = c1b.SUBMISSION.build_optimizer(ablation, OptimizerSpec(60, "cpu"))
        self.assertEqual(
            [group["lr"] for group in parent_bundle.optimizer.param_groups],
            [group["lr"] for group in ablation_bundle.optimizer.param_groups],
        )

    def test_train_and_eval_forward_values_and_execution_are_bitwise_equal(self):
        parent, ablation = self.paired_models()
        rows = prompts()
        for training in (True, False):
            parent.train(training); ablation.train(training)
            with torch.set_grad_enabled(training):
                parent_logits, parent_aux = parent(rows)
                ablation_logits, ablation_aux = ablation(rows)
            self.assertTrue(torch.equal(parent_logits, ablation_logits))
            self.assertTrue(torch.equal(parent_aux["active_updates"], ablation_aux["active_updates"]))
            self.assertEqual(parent_aux["executed_macrosteps"].item(), 3)
            self.assertEqual(ablation_aux["executed_macrosteps"].item(), 3)
        self.assertTrue(torch.all((ablation_aux["digit_probabilities"] == 0) | (ablation_aux["digit_probabilities"] == 1)))

    def parameter_gradient(self, model, row, target):
        model.train(); model.zero_grad(set_to_none=True)
        logits, _ = model(row)
        F.cross_entropy(logits[0, 6:9], target).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    def test_single_row_parameter_gradients_scale_exactly(self):
        labels = [torch.tensor([9, 9, 12]), torch.tensor([8, 10, 14]), torch.tensor([9, 16, 7])]
        for index, expected_scale in enumerate((1.0, 0.01, 0.01)):
            parent, ablation = self.paired_models()
            row = prompts()[index:index + 1]

            parent_logits, _ = parent(row)
            parent_logits.retain_grad()
            F.cross_entropy(parent_logits[0, 6:9], labels[index]).backward()
            parent_logit_gradient = parent_logits.grad.clone()
            ablation_logits, ablation_aux = ablation(row)
            ablation_aux["ungated_logits"].retain_grad()
            F.cross_entropy(ablation_logits[0, 6:9], labels[index]).backward()
            ablation_logit_gradient = ablation_aux["ungated_logits"].grad.clone()
            torch.testing.assert_close(
                ablation_logit_gradient,
                parent_logit_gradient * expected_scale,
                atol=2e-7,
                rtol=2e-5,
            )

            parent_gradient = self.parameter_gradient(parent, row, labels[index])
            ablation_gradient = self.parameter_gradient(ablation, row, labels[index])
            self.assertGreater(ablation_gradient.abs().sum().item(), 0)
            torch.testing.assert_close(
                ablation_gradient,
                parent_gradient * expected_scale,
                atol=2e-7,
                rtol=2e-5,
            )

    def test_mixed_loss_has_no_cross_row_gradient_contamination(self):
        _, ablation = self.paired_models()
        target = torch.tensor([9, 9, 12])
        alone = self.parameter_gradient(ablation, prompts()[0:1], target)
        ablation.zero_grad(set_to_none=True)
        mixed_logits, _ = ablation(prompts())
        F.cross_entropy(mixed_logits[0, 6:9], target).backward()
        mixed = torch.cat([p.grad.flatten() for p in ablation.parameters() if p.grad is not None])
        torch.testing.assert_close(alone, mixed, atol=2e-6, rtol=2e-6)

    def test_cpu_bfloat16_and_eval_hard_feedback(self):
        model = c1b.build_model(SPEC).train()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits, _ = model(prompts()[1:2])
            loss = F.cross_entropy(logits[0, 6:9], torch.tensor([8, 10, 14]))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
        model.eval()
        with torch.no_grad():
            _, auxiliary = model(prompts())
        self.assertTrue(torch.all((auxiliary["digit_probabilities"] == 0) | (auxiliary["digit_probabilities"] == 1)))


if __name__ == "__main__":
    unittest.main()
