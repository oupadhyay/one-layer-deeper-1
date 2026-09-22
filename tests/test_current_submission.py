"""Semantic tests for the active K6 candidate and experimental register paths."""

import unittest
import torch
from submission import (
    Model,
    ModelSpec,
    build_model,
    PAD_TOKEN_ID,
    BOS_TOKEN_ID,
    N_TOKEN_ID,
    X_TOKEN_ID,
    T_TOKEN_ID,
    ANS_TOKEN_ID,
    EOS_TOKEN_ID,
    DIGIT_TOKEN_OFFSET,
    USE_NEURAL_GPU,
    CELL_DIM,
    LOCAL_RADIUS,
    LOCAL_MICROTICKS,
    REGISTER_PLACES,
    STATE_TAIL_PLACES,
    USE_PAIRWISE_GRID,
    USE_SOFT_DIGIT_REGISTER,
    CrossAttention,
    LocalConvGRUCell,
)


class TestSubmissionSemantics(unittest.TestCase):
    def test_neural_gpu_shape(self) -> None:
        self.assertTrue(USE_NEURAL_GPU)
        self.assertEqual(CELL_DIM, 128)
        self.assertEqual(LOCAL_RADIUS, 2)
        self.assertEqual(LOCAL_MICROTICKS, 32)

    def test_token_constants(self) -> None:
        self.assertEqual(PAD_TOKEN_ID, 0)
        self.assertEqual(BOS_TOKEN_ID, 1)
        self.assertEqual(N_TOKEN_ID, 2)
        self.assertEqual(X_TOKEN_ID, 3)
        self.assertEqual(T_TOKEN_ID, 4)
        self.assertEqual(ANS_TOKEN_ID, 5)
        self.assertEqual(EOS_TOKEN_ID, 6)
        self.assertEqual(DIGIT_TOKEN_OFFSET, 7)

    def test_padded_batch_vectorized_lsb_features(self) -> None:
        # Padded Mixed-Length Batch:
        # Prompt 0: N 3 2 3 X 1 5 T 2 ANS 0 4 (len 12)
        # Prompt 1: N 1 1 X 7 T 1 ANS 9 PAD PAD (len 12, padded with 0)
        # Token IDs for Prompt 1:
        # N=2, '1'=8, '1'=8
        # X=3, '7'=14
        # T=4, '1'=8
        # ANS=5, '9'=16
        # PAD=0, PAD=0
        input_ids = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 9, 5, 7, 11],
            [2, 8, 8, 3, 14, 4, 8, 5, 16, 0, 0, 0],
        ])

        roles, places = Model.compute_vectorized_lsb_features(input_ids)

        expected_roles = torch.tensor([
            [0, 1, 1, 1, 0, 2, 2, 0, 3, 0, 4, 4],
            [0, 1, 1, 0, 2, 0, 3, 0, 4, 0, 0, 0],
        ])
        expected_places = torch.tensor([
            [0, 2, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ])

        self.assertTrue(
            torch.equal(roles, expected_roles),
            f"Padded batch roles mismatch!\nGot:\n{roles}\nExpected:\n{expected_roles}"
        )
        self.assertTrue(
            torch.equal(places, expected_places),
            f"Padded batch places mismatch!\nGot:\n{places}\nExpected:\n{expected_places}"
        )

    def test_curriculum_phase_boundaries(self) -> None:
        self.assertEqual(Model.curriculum_max_time_steps(1), 1)
        self.assertEqual(Model.curriculum_max_time_steps(6000), 1)
        self.assertEqual(Model.curriculum_max_time_steps(6001), 2)
        self.assertEqual(Model.curriculum_max_time_steps(8000), 2)
        self.assertEqual(Model.curriculum_max_time_steps(8001), 3)
        self.assertEqual(Model.curriculum_max_time_steps(9500), 3)

    @unittest.skipIf(USE_NEURAL_GPU, "stationary curriculum counter only")
    def test_curriculum_counter_changes_only_during_training(self) -> None:
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec)
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        model.eval()
        with torch.no_grad():
            model(input_ids, attention_mask=mask)
        self.assertEqual(model.curriculum_training_step.item(), 0)

        model.train()
        model(input_ids, attention_mask=mask)
        self.assertEqual(model.curriculum_training_step.item(), 1)

        model.eval()
        with torch.no_grad():
            model(input_ids, attention_mask=mask)
        self.assertEqual(model.curriculum_training_step.item(), 1)

    def test_curriculum_gradient_gate_preserves_values_and_scales_gradients(self) -> None:
        logits = torch.randn(2, 3, 17, requires_grad=True)
        parsed_time_steps = torch.tensor([1, 3])
        gated = Model.apply_curriculum_gradient_gate(
            logits, parsed_time_steps, maximum_time_steps=1
        )

        torch.testing.assert_close(gated, logits)
        gated.sum().backward()
        torch.testing.assert_close(logits.grad[0], torch.ones_like(logits.grad[0]))
        torch.testing.assert_close(
            logits.grad[1], torch.full_like(logits.grad[1], 0.01)
        )

    @unittest.skipIf(USE_NEURAL_GPU or USE_PAIRWISE_GRID or USE_SOFT_DIGIT_REGISTER, "prompt-position candidate only")
    def test_state_slots_have_distinct_tail_identity(self) -> None:
        spec = ModelSpec(vocab_size=20, max_seq_len=64, maximum_model_state_elements=500_000_000)
        model = build_model(spec)
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        roles, places = model.compute_vectorized_lsb_features(input_ids)
        _, state, state_mask = model.build_prompt_state(
            input_ids, roles, places, mask
        )
        tail_places, _ = model.compute_state_tail_places(input_ids, mask)
        self.assertEqual(tail_places[0, -2:].tolist(), [1, 0])
        self.assertTrue(torch.all(state_mask))
        self.assertGreater((state[0, -2] - state[0, -1]).abs().max().item(), 1e-6)

    @unittest.skipIf(USE_NEURAL_GPU or USE_PAIRWISE_GRID or USE_SOFT_DIGIT_REGISTER, "prompt-position candidate only")
    def test_padding_and_batch_composition_do_not_change_valid_logits(self) -> None:
        torch.manual_seed(74)
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec).eval()
        prompt = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
        padded = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 9, 0, 0, 0],
            [2, 10, 9, 10, 3, 8, 12, 13, 3, 8, 4, 9],
        ])
        padded_mask = padded != PAD_TOKEN_ID

        with torch.no_grad():
            solo, _ = model(prompt, attention_mask=prompt_mask)
            batched, _ = model(padded, attention_mask=padded_mask)
        torch.testing.assert_close(solo[0], batched[0, : prompt.shape[1]], atol=1e-5, rtol=1e-5)

    @unittest.skipIf(USE_NEURAL_GPU or USE_PAIRWISE_GRID or USE_SOFT_DIGIT_REGISTER, "prompt-position candidate only")
    def test_tail_logits_are_not_structurally_identical(self) -> None:
        torch.manual_seed(74)
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec).eval()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool))
        self.assertGreater((logits[0, -2] - logits[0, -1]).abs().max().item(), 1e-5)

    @unittest.skipUnless(USE_PAIRWISE_GRID, "pairwise candidate only")
    def test_pairwise_register_alignment_and_forward(self) -> None:
        spec = ModelSpec(vocab_size=20, max_seq_len=64, maximum_model_state_elements=500_000_000)
        model = build_model(spec)
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        residue, modulus, mask = model.build_digit_registers(
            input_ids, roles, places
        )

        expected_x = torch.tensor([12, 8, 7, 7])
        expected_n = torch.tensor([10, 9, 10, 7])
        place_embedding = model.register_place_embedding.weight
        torch.testing.assert_close(
            residue[0], model.token_embedding(expected_x) + place_embedding
        )
        torch.testing.assert_close(
            modulus[0], model.token_embedding(expected_n) + place_embedding
        )
        self.assertEqual(mask.tolist(), [[True, True, True, False]])
        logits, _ = model(input_ids)
        self.assertEqual(logits.shape, (1, input_ids.shape[1], 20))

    @unittest.skipUnless(USE_PAIRWISE_GRID, "pairwise candidate only")
    def test_pairwise_capacity_guard(self) -> None:
        spec = ModelSpec(vocab_size=20, max_seq_len=64, maximum_model_state_elements=500_000_000)
        model = build_model(spec)
        input_ids = torch.tensor([[
            N_TOKEN_ID,
            *([7] * (REGISTER_PLACES + 1)),
            X_TOKEN_ID,
            7,
            T_TOKEN_ID,
            8,
        ]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        with self.assertRaises(AssertionError):
            model.build_digit_registers(input_ids, roles, places)

    @unittest.skipUnless(USE_SOFT_DIGIT_REGISTER, "soft-digit candidate only")
    def test_soft_digit_register_alignment_and_forward(self) -> None:
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec).eval()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        residue, modulus, mask = model.build_digit_registers(input_ids, roles, places)

        expected_x = torch.tensor([12, 8, 7, 7])
        expected_n = torch.tensor([10, 9, 10, 7])
        place_identity = model.register_place_embedding.weight
        torch.testing.assert_close(
            residue[0], model.token_embedding(expected_x) + place_identity
        )
        torch.testing.assert_close(
            modulus[0], model.token_embedding(expected_n) + place_identity
        )
        self.assertEqual(mask.tolist(), [[True, True, True, False]])
        with torch.no_grad():
            logits, _ = model(input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool))
        self.assertEqual(logits.shape, (1, input_ids.shape[1], 17))
        self.assertGreater((logits[0, -2] - logits[0, -1]).abs().max().item(), 1e-5)

    @unittest.skipUnless(USE_SOFT_DIGIT_REGISTER, "soft-digit candidate only")
    def test_soft_requantization_is_differentiable_probability_feedback(self) -> None:
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec)
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        residue, modulus, mask = model.build_digit_registers(input_ids, roles, places)
        digit_basis = model.token_embedding.weight[7:17]

        updated, digit_logits = model.soft_digit_transition(
            residue, modulus, mask, digit_basis, model.register_place_embedding.weight
        )
        expected = digit_logits.softmax(dim=-1) @ digit_basis
        expected = expected + model.register_place_embedding.weight[None, :, :]
        torch.testing.assert_close(updated[mask], expected[mask])
        self.assertTrue(updated.requires_grad)

    @unittest.skipUnless(USE_SOFT_DIGIT_REGISTER, "soft-digit candidate only")
    def test_soft_candidate_is_padding_and_batch_invariant(self) -> None:
        torch.manual_seed(74)
        spec = ModelSpec(vocab_size=17, max_seq_len=16, maximum_model_state_elements=500_000_000)
        model = build_model(spec).eval()
        prompt = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 9]])
        padded = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 9, 0, 0, 0],
            [2, 10, 9, 10, 3, 8, 12, 13, 3, 8, 4, 9],
        ])
        with torch.no_grad():
            solo, _ = model(prompt, attention_mask=torch.ones_like(prompt, dtype=torch.bool))
            batched, _ = model(padded, attention_mask=padded != PAD_TOKEN_ID)
        torch.testing.assert_close(solo[0], batched[0, : prompt.shape[1]], atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(USE_NEURAL_GPU, "Neural-GPU candidate only")
    def test_tape_register_alignment_and_zero_extension(self) -> None:
        model = build_model(ModelSpec(17, 16, 500_000_000))
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        residue = model.extract_lsd_digit_register(input_ids, roles, places, 2)
        modulus = model.extract_lsd_digit_register(input_ids, roles, places, 1)
        self.assertEqual(residue.tolist(), [[5, 1, 0, 0]])
        self.assertEqual(modulus.tolist(), [[3, 2, 3, 0]])

    @unittest.skipUnless(USE_NEURAL_GPU, "Neural-GPU candidate only")
    def test_local_cell_receptive_field(self) -> None:
        torch.manual_seed(74)
        cell = LocalConvGRUCell(16, radius=2).eval()
        baseline = torch.randn(1, 16, 16)
        perturbed = baseline.clone()
        perturbed[:, 8] += 1.0
        with torch.no_grad():
            delta = (cell(perturbed) - cell(baseline)).abs().amax(dim=-1)[0]
        outside = torch.cat((delta[:6], delta[11:]))
        torch.testing.assert_close(outside, torch.zeros_like(outside), atol=0, rtol=0)

    @unittest.skipUnless(USE_NEURAL_GPU, "Neural-GPU candidate only")
    def test_neural_gpu_padding_and_batch_invariance(self) -> None:
        torch.manual_seed(74)
        model = build_model(ModelSpec(17, 16, 500_000_000)).eval()
        prompt = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8]])
        padded = torch.tensor([
            [2, 10, 9, 10, 3, 8, 12, 4, 8, 0, 0, 0],
            [2, 10, 9, 10, 3, 8, 12, 13, 3, 8, 4, 8],
        ])
        with torch.no_grad():
            solo, _ = model(prompt, attention_mask=torch.ones_like(prompt, dtype=torch.bool))
            batched, _ = model(padded, attention_mask=padded != PAD_TOKEN_ID)
        torch.testing.assert_close(solo[0], batched[0, : prompt.shape[1]], atol=1e-5, rtol=1e-5)

    @unittest.skipUnless(USE_NEURAL_GPU, "Neural-GPU candidate only")
    def test_shared_cell_scratch_reset_and_feedback_gradient(self) -> None:
        model = build_model(ModelSpec(17, 16, 500_000_000))
        cells = [module for module in model.modules() if isinstance(module, LocalConvGRUCell)]
        self.assertEqual(len(cells), 1)
        residue = torch.randn(1, REGISTER_PLACES, CELL_DIM, requires_grad=True)
        modulus = torch.tensor([[3, 2, 3, 0]])
        tape1, _ = model.build_local_tape(residue, modulus)
        tape2, _ = model.build_local_tape(residue + 1.0, modulus)
        torch.testing.assert_close(tape1[:, 4:8], tape2[:, 4:8])

        next_residue, _, _ = model.run_local_macrostep(residue, modulus)
        _, second_logits, _ = model.run_local_macrostep(next_residue, modulus)
        gradient = torch.autograd.grad(second_logits.sum(), residue)[0]
        self.assertGreater(gradient.abs().sum().item(), 0.0)

    @unittest.skipUnless(USE_NEURAL_GPU, "Neural-GPU candidate only")
    def test_macrostep_decoder_is_final_endpoint_decoder(self) -> None:
        torch.manual_seed(74)
        model = build_model(ModelSpec(17, 16, 500_000_000)).eval()
        input_ids = torch.tensor([[2, 10, 9, 10, 3, 8, 12, 4, 8]])
        roles, places = model.compute_vectorized_lsb_features(input_ids)
        residue_digits = model.extract_lsd_digit_register(input_ids, roles, places, 2)
        modulus_digits = model.extract_lsd_digit_register(input_ids, roles, places, 1)
        with torch.no_grad():
            _, macro_logits, _ = model.run_local_macrostep(
                model.digit_embedding(residue_digits), modulus_digits
            )
            endpoint, _ = model(input_ids, attention_mask=torch.ones_like(input_ids, dtype=torch.bool))
        answer_places = model.compute_answer_places(input_ids, torch.ones_like(input_ids, dtype=torch.bool))
        expected = torch.gather(
            macro_logits,
            1,
            answer_places[:, :, None].expand(-1, -1, 10),
        )
        torch.testing.assert_close(endpoint[..., 7:17], expected)

    def test_cross_attention_context_masking(self) -> None:
        cross_attn = CrossAttention(dim=16, num_heads=2)
        state = torch.randn(1, 4, 16)
        context1 = torch.randn(1, 6, 16)
        context2 = context1.clone()
        # Modify context2 at masked positions (positions 4 and 5)
        context2[:, 4:, :] = torch.randn(1, 2, 16) * 100.0

        # Context mask: True for positions 0..3, False for positions 4..5
        context_mask = torch.tensor([[True, True, True, True, False, False]])

        out1 = cross_attn(state, context1, context_mask)
        out2 = cross_attn(state, context2, context_mask)

        # Outputs MUST be identical despite context2 being completely different at masked positions!
        self.assertTrue(
            torch.allclose(out1, out2, atol=1e-5),
            "CrossAttention output changed when masked context values were modified!"
        )


if __name__ == "__main__":
    unittest.main()
