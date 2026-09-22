import json
import re
from datetime import datetime
from pathlib import Path
import unittest


ROOT = Path(__file__).parents[1]
PORTAL = ROOT / "research_portal"
EXPECTED = {
    "latent_exposure_singlethread_v1", "latent_inner_refinement_x2_singlethread_v1",
    "latent_x2_t1grad_singlethread_v1", "latent_x2_t1grad_d64_singlethread_v1",
    "biscan_t1grad_singlethread_v1", "biscan_e5_diversity_singlethread_v1",
    "relative_place_attention_t1grad_singlethread_v1", "reversible_ssm_t1grad_singlethread_v1",
    "direct_transformer_singlethread_v1", "digit_pair_cellular_singlethread_v1",
    "factorized_pair_reducer_singlethread_v1_trainability400",
    "factorized_pair_reducer_d96k3_singlethread_v1_trainability400",
    "compressed_axial_cellular_singlethread_v1_trainability400",
    "eight_thread_replay_bitwise_gate",
    "remote_easy_e5_axial_large_v1_e1ad445e",
    "remote_easy_e5_axial_safe_batch_dfaf1f08",
    "remote_easy_e5_axial_cuda_safe_c2bb2d1f",
    "remote_easy_e5_axial_smoke_03dfb525",
    "remote_easy_e5_baseline_control_7bd61fbd",
    "remote_easy_e5_parser_bisect_17e40a97",
    "remote_easy_e5_workspace_bisect_3be7fc6e",
    "remote_easy_e5_one_h_axial_cell_893312cf",
    "remote_easy_e5_four_calls_14c93f32",
    "remote_easy_e5_boundary_readout_65013c72",
    "remote_easy_e5_full_forward_smoke_10edb4ca",
    "remote_easy_e5_axial_v5_40d17e44",
    "remote_easy_e5_axial_v6_batch512_589bfa67",
    "remote_easy_e5_compressed_axial_v7_99de059b",
    "remote_easy_e5_compressed_axial_v8_d685bb29",
    "remote_easy_e5_compressed_axial_v9_d3cb7070",
    "remote_easy_e5_compressed_axial_v10_b236863b",
    "remote_easy_e5_compressed_axial_v11_10978ea6",
    "remote_easy_e5_compressed_axial_v12_c2628b71",
    "remote_easy_e5_compressed_axial_v13_850711cb",
    "remote_easy_e5_compressed_axial_v13b_20ecba00",
    "remote_easy_e5_compressed_axial_v14_bee81b5f",
    "remote_easy_e5_compressed_axial_v15_dd7e400d",
    "remote_easy_e5_compressed_axial_v16_3aecdf84",
    "remote_easy_e5_factorized_pair_v1_d8d9669e",
    "remote_easy_e5_compressed_axial_v17_1c1df927",
    "remote_easy_e5_factorized_pair_v2_77b5ac0a",
    "remote_easy_e5_factorized_pair_v3_e5c541cd",
    "remote_easy_e5_factorized_pair_v4_50df29fc",
    "remote_easy_e5_factorized_pair_v5_033aadba",
    "remote_easy_e5_factorized_pair_v6_c975324d",
    "remote_easy_e5_factorized_pair_v7_53f131d1",
    "remote_easy_e5_factorized_pair_v8_394d852b",
    "remote_easy_e5_factorized_pair_v9_395049ae",
    "remote_easy_e5_relational_transducer_v1_97011491",
    "remote_easy_e5_neural_gpu_v1_cf225723",
    "remote_easy_e5_n_operator_v1_87aa89f3",
    "remote_medium_m5_neural_gpu_v2_2539b938",
    "remote_easy_e1_biscan_v1_1ec512d3",
    "remote_easy_e1_biscan_v2_0d76dd56",
    "remote_medium_m3_biscan_v3_04ce7b55",
    "remote_easy_e5_neural_gpu_v3_63fa0c57",
    "remote_easy_e5_neural_gpu_v4_0c81a9a9",
    "remote_easy_e5_neural_gpu_v5_2bba0b15",
    "remote_easy_e5_tcet_v1_6eb2411f",
    "remote_medium_m3_neural_gpu_v2_b348d499",
    "remote_easy_e5_urt_v1_ff0a36f5",
    "remote_easy_e5_urt_v2_57ecceb7",
    "remote_easy_e5_urt_v3_ef94b31c",
    "remote_easy_e5_urt_v4_1e239227",
    "remote_easy_e5_urt_v5_c09d117d",
    "remote_easy_e5_urt_v6_c5c872f1",
    "remote_easy_e5_urt_v7_89c9e6ba",
    "remote_easy_e5_autoregressive_v1_fd1cae7a",
    "remote_easy_e5_autoregressive_v1_12cba492",
    "remote_easy_e5_dnc_v1_3c4556e2",
    "remote_easy_e5_deq_v1_0b74e09d",
    "remote_easy_e5_deq_v1_52ffedf3",
    "remote_easy_e5_dual_path_v1_4be20766",
    "remote_easy_e5_identity_dynamic_v1_064cbf80",
    "remote_easy_e5_pbep_v1_37612a20",
    "remote_easy_e5_ntt_v1_bb058571",
    "whole_output_energy_easy_v1_resource_abort",
    "remote_easy_e5_fnst_v1_1e1c0aba",
    "remote_easy_e5_wfst_v1_5ed99a33",
    "deltanet_fast_weight_easy_v1_resource_abort",
    "tpr_bilinear_easy_v1_resource_abort",
    "rim_transition_easy_v1_resource_abort",
    "hyena_register_easy_v1_resource_abort",
    "remote_easy_e5_xca_v1_ec2642fb",
    "remote_easy_e5_sgu_v1_b5400631",
    "remote_easy_e5_joint_mlp_v1_28946133",
    "sgu_register_medium_m5_v1_resource_abort",
    "remote_medium_m3_sgu_v1_50ead117",
    "remote_easy_e5_sgu_latent_v1_575e10fc",
    "remote_easy_e5_sgu_highway_v1_253cfc0c",
    "remote_easy_e5_sgu_amlp_v1_82a1dece",
    "remote_easy_e5_sgu_ste_v1_b7d46ab2",
    "remote_easy_e5_sgu_axial_v1_b6056bbd",
    "remote_easy_e5_sgu_rms_v1_f0548ca2",
    "remote_easy_e5_sgu_amlp_rms_v1_7a47f0ec",
    "remote_easy_e5_sgu_amlp_conv_v1_0b02a667",
    "remote_easy_e5_sgu_highway_rms_v1_2c7bdfcf",
    "remote_easy_e5_sgu_highway_conv_v1_4a96dcd9",
    "remote_medium_m3_sgu_amlp_v1_427f9bbf",
    "remote_easy_e5_sgu_rms_functional_v1_8e15cfea",
    "remote_easy_e5_sgu_axial_einsum_v1_e7fb6b40",
    "remote_easy_e5_sgu_rms_native_v1_a6fca47a",
    "remote_easy_e5_sgu_axial_role_place_v1_283e5fe2",
    "remote_easy_e5_sgu_rms_input_ln_v1_4ef89339",
    "remote_easy_e5_sgu_axial_place_role_v1_4aa81dd9",
    "remote_easy_e5_sgu_rms_gate_ln_v1_7b7fd906",
    "remote_easy_e5_sgu_axial_serial_v1_81292d40",
    "remote_easy_e5_sgu_rms_readout_ln_v1_99292a35",
    "remote_easy_e5_sgu_axial_second_order_v1_d86ebf4e",
    "remote_easy_e5_sgu_rms_eps_v1_958cfa16",
    "remote_easy_e5_sgu_axial_rank1_v1_e20a758a",
    "remote_easy_e5_sgu_cross_c11_v1_cf7e0b91",
    "remote_easy_e5_sgu_cross_c12_v1_2fe53acc",
    "remote_easy_e5_sgu_cross_c21_v1_4ffda9dc",
    "remote_easy_e5_sgu_cross_c22_v1_6e41c02d",
    "remote_easy_e5_sgu_cross_c33_v1_c30e74c8",
    "remote_easy_e5_sgu_cross_c34_v1_7b380ccd",
    "remote_easy_e5_sgu_cross_c43_v1_e83e7684",
    "remote_easy_e5_sgu_cross_c44_v1_1105c965",
    "remote_easy_e5_sgu_cal_c21_v1_fb3aa63d",
    "remote_easy_e5_sgu_cal_c11_v1_6d709b14",
    "remote_easy_e5_sgu_cal_c12_v1_8fe5a1ee",
    "remote_easy_e5_sgu_cal_c33_v1_7490b20a",
    "remote_easy_e5_sgu_cal_c44_v1_21e68712",
    "remote_easy_e5_sgu_cal_c34_v1_96bcc37c",
    "remote_easy_e5_sgu_diag_c21_v1_c21bb935",
    "remote_easy_e5_sgu_diag_c11_v1_ae9fd212",
    "remote_easy_e5_sgu_diag_calc12_v1_abfba39f",
    "remote_easy_e5_sgu_diag_calc33_v1_dfa64e95",
    "remote_easy_e5_sgu_bias_c21_v1_1af35559",
    "remote_easy_e5_sgu_bias_diagcalc12_v1_82ed4b45",
    "remote_easy_e5_sgu_bias_diagcalc33_v1_2e5c1e8f",
    "remote_easy_e5_sgu_unorm_c21_v1_cc112187",
    "remote_easy_e5_sgu_unorm_diagcalc12_v1_7df27775",
    "remote_easy_e5_sgu_opt_lr4_v1_701bce0e", "remote_easy_e5_sgu_opt_lr8_v1_1214f695",
    "remote_easy_e5_sgu_opt_warm16_v1_1ebe5348", "remote_easy_e5_sgu_opt_warm64_v1_a75f6877",
    "remote_easy_e5_sgu_opt_batch128_v1_d8109afb", "remote_easy_e5_sgu_opt_batch512_v1_78341c53",
    "remote_easy_e5_sgu_opt_nodecay_v1_5960dec3", "remote_easy_e5_sgu_opt_h168_v1_60101094",
    "remote_easy_e5_sgu_opt_h168_v1_b2a218d7", "remote_easy_e5_sgu_opt_h280_v1_e5336b0d",
    "remote_easy_e5_sgu_opt_init1e3_v1_b954c42f",
    "remote_medium_m3_sgu_medium_rms_dense_v1_5b228033",
    "remote_medium_m3_sgu_medium_axial_ln_v1_0e90e3e6",
    "remote_medium_m3_sgu_medium_c21_v1_5432b91d",
    "remote_medium_m3_sgu_medium_c21_h280_v1_ea0d3447",
    "remote_medium_m3_sgu_axial_nodecay_final_v1_7522f330",
    "remote_easy_e5_sgu_relaxed_feedback_v1_f2a54758",
    "remote_easy_e5_sgu_toeplitz_v1_5125bd47",
    "remote_easy_e5_sgu_attention_v1_fb416171",
    "remote_easy_e5_sgu_biscan_v1_fd5ac455",
    "remote_medium_m3_sgu_relaxed_v1_f578dffc",
    "remote_hard_h1_dynamic_biscan_v1_d2e6b02a",
    "remote_easy_e3_conditional_cross_v1_16b3d275",
    "remote_easy_e1_conditional_cross_v1_c3844fa8",
    "remote_easy_e3_conditional_shared_v2_50d94553",
    "remote_easy_e3_conditional_global_v1_390e8647",
    "remote_easy_e5_conditional_global_v1_b23b9429",
    "remote_easy_e5_conditional_global_batch256_v2_80ef984c",
    "remote_easy_e5_conditional_opt_lr3_v1_66cabb7a_failure",
    "remote_easy_e5_conditional_opt_lr4_v1_b8978a7c_failure",
    "remote_easy_e5_conditional_control_v2_3833f72f_failure",
    "remote_easy_e5_conditional_opt_lr3_v2_1182b32b",
    "remote_easy_e5_conditional_optimizer1_lr2_v1_58d91768",
    "remote_easy_e5_conditional_optimizer1_lr4_v1_d8721da9",
    "remote_easy_e5_conditional_optimizer1_lr5_v1_3d399107",
    "remote_easy_e5_conditional_optimizer1_lr7_v1_7a77d963",
    "remote_easy_e5_conditional_optimizer1_lr8_v1_12f2232a",
    "remote_easy_e5_conditional_optimizer1_lr10_v1_010e1c66",
    "remote_easy_e5_conditional_optimizer1_lr12_v1_8b9bb974",
    "remote_easy_e5_conditional_optimizer1_lr15_v1_4a8566f5",
    "remote_easy_e5_conditional_optimizer1_wd0_v1_b5a69764",
    "remote_easy_e5_conditional_optimizer1_wd3_v1_823ecb79",
    "remote_easy_e5_conditional_optimizer1_wd10_v1_493208b4",
    "remote_easy_e5_conditional_optimizer1_warm8_v1_eef35e1c",
    "remote_easy_e5_conditional_optimizer1_warm16_v1_ad84e2fa",
    "remote_easy_e5_conditional_optimizer1_warm64_v1_59f070c0",
    "remote_easy_e5_conditional_optimizer1_batch128_v1_16dd02da",
    "remote_easy_e5_conditional_optimizer2_lr9_v1_5a4fd253",
    "remote_easy_e5_conditional_optimizer2_lr11_v1_41e445c2",
    "remote_easy_e5_conditional_optimizer2_lr10_batch128_v1_c61874f3",
    "remote_easy_e5_conditional_optimizer2_lr10_batch512_v1_ae7a2723",
    "remote_easy_e5_conditional_optimizer2_lr10_wd0_v1_9e57e5e7",
    "remote_easy_e5_conditional_optimizer2_lr10_wd3_v1_45577916",
    "remote_easy_e5_conditional_optimizer2_lr10_wd10_v1_91151ae3",
    "remote_easy_e5_conditional_optimizer2_lr10_warm8_v1_0d5318fc",
    "remote_easy_e5_conditional_optimizer2_lr10_warm16_v1_174a7d5f",
    "remote_easy_e5_conditional_optimizer2_lr10_warm64_v1_15f79149",
    "remote_easy_e5_conditional_optimizer3_warm8_lr8_v1_f4732893",
    "remote_easy_e5_conditional_optimizer3_warm8_lr9_v1_56392352",
    "remote_easy_e5_conditional_optimizer3_warm8_lr11_v1_903519a5",
    "remote_easy_e5_conditional_optimizer3_warm8_lr12_v1_cf3e0c1d",
    "remote_easy_e5_conditional_optimizer3_warm8_batch128_v1_b18c3f7a",
    "remote_easy_e5_conditional_optimizer3_warm8_batch512_v1_941090be",
    "remote_easy_e5_conditional_optimizer3_warm8_wd0_v1_c0ca6809",
    "remote_easy_e5_conditional_optimizer3_warm8_wd3_v1_a5158b32",
    "remote_easy_e5_conditional_optimizer3_warm8_wd10_v1_3a954627",
    "remote_easy_e5_conditional_optimizer3_warm8_beta98_v1_77a257ed",
    "remote_easy_e5_conditional_loss1_smooth02_v1_d072c047",
    "remote_easy_e5_conditional_loss1_smooth05_v1_f2074d30",
    "remote_easy_e5_conditional_loss1_smooth10_v1_40f738fb",
    "remote_easy_e5_conditional_loss1_focal05_v1_72043c1e",
    "remote_easy_e5_conditional_loss1_focal10_v1_dbf83aea",
    "remote_easy_e5_conditional_loss1_focal20_v1_14afe792",
    "remote_easy_e5_conditional_loss1_hard25_v1_49d8761e",
    "remote_easy_e5_conditional_loss1_hard50_v1_2a45f78a",
    "remote_easy_e5_conditional_loss2_focal075_v1_e9b0c9ba",
    "remote_easy_e5_conditional_loss2_focal125_v1_f0de8615",
    "remote_easy_e5_conditional_loss2_focal150_v1_f63f6e33",
    "remote_easy_e5_conditional_loss2_hard35_v1_d832f703",
    "remote_easy_e5_conditional_loss2_hard75_v1_49a9e08e",
    "remote_easy_e5_conditional_loss2_hard100_v1_e7103421",
    "remote_easy_e5_conditional_loss2_focal05_lr9_v1_5b5b2568",
    "remote_easy_e5_conditional_loss2_focal05_lr11_v1_aca84347",
    "remote_easy_e5_conditional_loss2_focal05_batch128_v1_6b9d4952",
    "remote_easy_e5_conditional_loss2_focal05_batch512_v1_674df245",
    "remote_easy_e5_conditional_loss2_focal05_wd0_v1_68306aaa",
    "remote_easy_e5_conditional_loss2_focal05_warm16_v1_68968ed9",
    "remote_easy_e5_conditional_loss-repro_focal10_repro_v1_b8ac941e",
    "remote_medium_m3_conditional_medium_focal05_lr9_v1_a2c10fbb",
    "remote_medium_m3_conditional_medium_plain_lr9_v1_01a87096",
    "remote_medium_m3_conditional_medium_focal10_lr9_v1_0ef94ca2",
    "remote_medium_m3_conditional_medium_focal05_lr6_v1_96b4b8c3",
    "remote_medium_m3_conditional_medium_focal05_batch128_v1_4b7c196c",
    "remote_medium_m3_conditional_medium_focal05_d96_v1_193e56df",
    "remote_medium_m3_conditional_medium_focal05_lr10_v1_e4bf7a30",
    "remote_medium_m3_conditional_medium_focal05_warm32_v1_409b9f8d",
    "remote_medium_m3_conditional_medium_focal05_batch512_v1_499481fd",
    "remote_medium_m3_conditional_medium_focal05_d128_v1_d7b4d509",
    "remote_hard_h1_conditional_dynamic_hard_v1_a465096d",
    "remote_easy_e5_conditional_t1_weight2_v1_d962bde3",
    "remote_easy_e5_conditional_t1_weight4_v1_275c24a9_failure",
    "remote_easy_e5_conditional_t1_weight4_plain_v1_95c15645",
    "remote_easy_e5_conditional_t1_weight4_d64_v1_ff1b4bdc",
    "remote_easy_e5_conditional_t1_weight8_v1_9e8b6928_failure",
    "remote_hard_h1_conditional_d64_v1_ac1e71f6",
    "remote_easy_e5_conditional_d64_pure_t1_weight2_v1_2eb9adc5",
    "remote_easy_e5_tpr_bilinear_v2_two_ticks_38f38c27",
    "remote_easy_e5_tpr_bilinear_v1_de7e0a23",
    "remote_easy_e5_deltanet_fast_weight_d64_v2_fe7b6e95",
    "remote_easy_e5_deltanet_fast_weight_d64_one_sweep_v3_0ed23134",
    "remote_easy_e5_deltanet_fast_weight_d64_three_sweep_v4_0c1f137c",
    "remote_easy_e5_agreement_capsule_v1_cd7a2ccf",
    "remote_easy_e5_hierarchical_probabilistic_circuit_v1_f7314a5c",
    "remote_easy_e5_soft_oblivious_forest_v1_d621cb72",
}


def registry():
    source = (PORTAL / "experiments.js").read_text(encoding="utf-8")
    match = re.fullmatch(r".*?const EXPERIMENTS = (\[.*\]);\s*", source, re.S)
    assert match, "registry must remain a clearly delimited JSON-compatible assignment"
    return json.loads(match.group(1)), source


def run_details():
    source = (PORTAL / "run_details.js").read_text(encoding="utf-8")
    match = re.fullmatch(r"const RUN_DETAILS = (\{.*\});\s*", source, re.S)
    assert match, "dashboard data must remain a clearly delimited JSON-compatible assignment"
    return json.loads(match.group(1))


class ResearchPortalTests(unittest.TestCase):
    def test_registry_has_exactly_the_two_hundred_forty_eight_frozen_records_once(self):
        records, source = registry()
        ids = [record["id"] for record in records]
        self.assertEqual(len(records), 248)
        self.assertEqual(set(ids), EXPECTED)
        self.assertTrue(all(ids.count(identifier) == 1 for identifier in EXPECTED))
        self.assertTrue(
            all(source.count(f'"id":"{identifier}"') == 1 for identifier in EXPECTED)
        )

    def test_required_fields_metrics_and_artifacts(self):
        records, _ = registry()
        required = {
            "id", "group", "title", "protocol", "verdict", "modelState",
            "updates", "examples", "metrics", "conclusion", "limitation", "links",
        }
        for record in records:
            with self.subTest(experiment=record["id"]):
                self.assertLessEqual(required, record.keys())
                self.assertTrue(record["conclusion"])
                self.assertTrue(record["limitation"])
                self.assertTrue(record["verdict"])
                if record.get("status"):
                    self.assertIn(record["status"], {"aborted", "failed"})
                    self.assertEqual(record["metrics"], {})
                elif record.get("stage") == "trainability":
                    self.assertLessEqual({"train", "trainT1"}, record["metrics"].keys())
                    for metric in record["metrics"].values():
                        self.assertEqual(set(metric), {"exact", "ce"})
                    self.assertEqual(set(record["links"]), {"result", "provenance"})
                elif record.get("stage") == "remote":
                    self.assertLessEqual(
                        {"train", "test", "ood", "oodNT1"},
                        record["metrics"].keys(),
                    )
                    for metric in record["metrics"].values():
                        self.assertEqual(set(metric), {"exact", "ce"})
                    self.assertEqual(set(record["links"]), {"result", "provenance"})
                elif record.get("stage") == "remote-diagnostic":
                    self.assertLessEqual({"train", "test", "ood"}, record["metrics"].keys())
                    for metric in record["metrics"].values():
                        self.assertEqual(set(metric), {"exact", "ce"})
                    self.assertEqual(set(record["links"]), {"result", "provenance"})
                else:
                    self.assertLessEqual(
                        {"train", "trainT1", "test", "testT1", "seenT1", "oodT6"},
                        record["metrics"].keys(),
                    )
                    for metric in record["metrics"].values():
                        self.assertEqual(set(metric), {"exact", "ce"})
                    self.assertEqual(set(record["links"]), {"result", "provenance"})
                for target in record["links"].values():
                    self.assertTrue((PORTAL / target).resolve().is_file(), target)

    def test_safe_rendering_and_page_wiring(self):
        index = (PORTAL / "index.html").read_text(encoding="utf-8")
        app = (PORTAL / "app.js").read_text(encoding="utf-8")
        self.assertIn('href="style.css"', index)
        self.assertIn('<script src="experiments.js"></script>', index)
        self.assertIn('<script src="run_details.js"></script>', index)
        self.assertIn('<script src="app.js"></script>', index)
        self.assertIn("Rung accuracy by run date", index)
        self.assertIn('id="seen-chart"', index)
        self.assertIn('id="ood-chart"', index)
        self.assertIn('id="rung-filter"', index)
        self.assertIn('id="run-dialog"', index)
        self.assertIn("textContent", app)
        self.assertIn("createElement", app)
        self.assertIn("showModal", app)
        self.assertIn('addEventListener("pointerenter"', app)
        self.assertIn('addEventListener("focus"', app)
        self.assertIn('addEventListener("click"', app)
        self.assertIn("function jointT1(exp)", app)
        self.assertIn("Math.min(seen.accuracy, ood.accuracy)", app)
        self.assertIn("Math.min(seen.accuracy / (4 / 512), ood.accuracy / (2 / 512))", app)

        self.assertIn("publicPass: transferRatio > 1", app)

        self.assertIn("function verdictCategory(exp)", app)
        self.assertIn('verdictCategory(exp) === verdictFilter.value', app)
        for category in ("Transfer success", "Partial success", "Failed gate", "Aborted / invalid", "Diagnostic / inconclusive"):

            self.assertIn(category, app)

        self.assertNotIn("EXPERIMENTS.map(x => x.verdict)", app)

        self.assertIn("All outcome categories", index)
        self.assertNotIn("innerHTML", app)
        self.assertNotIn("insertAdjacentHTML", app)
        self.assertNotIn("fetch(", app)
        self.assertNotIn("fetch(", index)

    def test_dashboard_details_cover_all_dated_remote_runs(self):
        details = run_details()
        self.assertEqual(len(details), 222)
        self.assertEqual(len(details), len(set(details)))
        for identifier, detail in details.items():
            with self.subTest(experiment=identifier):
                self.assertIn(detail["tier"], {"easy", "medium", "hard"})
                datetime.fromisoformat(detail["createdAt"].replace("Z", "+00:00"))
                self.assertTrue(detail["architecture"])
                for key in ("seenRungs", "oodRungs"):
                    time_steps = []
                    for rung in detail[key]:
                        self.assertEqual(
                            set(rung), {"timeSteps", "correct", "total"}
                        )
                        self.assertIn(rung["timeSteps"], {1, 2, 4, 8, 16, 32, 64})
                        self.assertGreater(rung["total"], 0)
                        self.assertGreaterEqual(rung["correct"], 0)
                        self.assertLessEqual(rung["correct"], rung["total"])
                        time_steps.append(rung["timeSteps"])
                    self.assertEqual(time_steps, sorted(set(time_steps)))


if __name__ == "__main__":
    unittest.main()
