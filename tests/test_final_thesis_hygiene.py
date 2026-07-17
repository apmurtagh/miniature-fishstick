import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "baselines" / "lgbm_numeric_v1_subsample"
DOCS = ROOT / "docs" / "thesis_final"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_readme_final_submission_pointer_is_current():
    readme = read_text(ROOT / "README.md")

    assert "proposal-gap-uplift-20260713_100844" in readme
    assert "thesis-final-v23-proxy-cohort-diagnostics-20260717" in readme
    assert "notebooks/governance_ready_fraud_decisioning_end_to_end_reproduction.ipynb" in readme
    assert "docs/thesis_final/final_submission_artifact_manifest.md" in readme
    assert "20,000-row constrained LLM robustness run" in readme
    assert "Completed 1,000-row inference-time feature-masking re-score" in readme


def test_claim_calibration_docs_exist_and_are_bounded():
    expected = {
        "rq_hypothesis_disposition.md": [
            "RQ3 Decision utility / simulatability",
            "Not directly tested",
            "H2 Constraint value",
            "future work",
        ],
        "validator_acceptance_policy.md": [
            "Clean accepted",
            "Accepted with validation flags",
            "Hard failure",
            "Audit-complete",
            "zero validation concern",
        ],
        "direction_proxy_interpretation.md": [
            "0.3282",
            "conservative rule-based review flag",
            "not as proof",
            "Full direction-level semantic faithfulness remains future work",
        ],
        "feature_masking_sample_rationale.md": [
            "1,000-row sample",
            "20,000 rows",
            "1.11e-16",
            "inference-time robustness stress test",
        ],
    }

    for filename, phrases in expected.items():
        path = DOCS / filename
        assert path.exists(), f"Missing {path}"
        text = read_text(path)
        for phrase in phrases:
            assert phrase in text, f"Missing phrase in {filename}: {phrase}"


def test_final_manifest_references_closure_and_governance_artifacts():
    manifest = read_text(DOCS / "final_submission_artifact_manifest.md")

    required = [
        "Final Closure and Claim-Calibration Artefacts",
        "rq_hypothesis_disposition.md",
        "validator_acceptance_policy.md",
        "direction_proxy_interpretation.md",
        "feature_masking_sample_rationale.md",
        "Final Low-Effort Governance Uplifts",
        "final_governance_controls_runbook.md",
        "semantic_validator_proxy.py",
    ]

    for phrase in required:
        assert phrase in manifest


def test_feature_masking_x_test_manifest_reproduces_original_predictions():
    manifest_path = ART / "model_ready_x_test_manifest.json"
    assert manifest_path.exists(), "Missing model_ready_x_test_manifest.json"

    manifest = read_json(manifest_path)

    assert manifest["status"] == "created"
    assert manifest["n_rows"]["test"] == 20000
    assert manifest["n_features"] == 40

    match = manifest["prediction_id_match"]
    assert match["same_length"] is True
    assert match["exact_order_match"] is True
    assert match["matched_positions"] == 20000

    repro = manifest["prediction_reproduction"]
    assert repro["close_at_1e_10"] is True
    assert repro["close_at_1e_6"] is True
    assert repro["max_abs_diff"] < 1e-10


def test_feature_masking_rescore_summary_is_present_and_bounded():
    summary_path = ART / "feature_masking_rescore" / "feature_masking_rescore_summary.json"
    assert summary_path.exists(), "Missing feature_masking_rescore_summary.json"

    summary = read_json(summary_path)

    assert summary["status"] == "run"
    assert summary["rows_sampled"] == 1000
    assert summary["mask_rates"] == [0.0, 0.3, 0.6]
    assert len(summary["summary_rows"]) == 3
    assert "controlled robustness stress test" in summary["caveat"]

    by_rate = {row["mask_rate"]: row for row in summary["summary_rows"]}
    assert by_rate[0.0]["action_change_rate"] == 0.0
    assert 0.015 <= by_rate[0.3]["action_change_rate"] <= 0.025
    assert 0.015 <= by_rate[0.6]["action_change_rate"] <= 0.03


def test_semantic_validator_proxy_summary_is_present_and_caveated():
    summary_path = ART / "semantic_validator_proxy" / "semantic_validator_proxy_summary.json"
    assert summary_path.exists(), "Missing semantic_validator_proxy_summary.json"

    summary = read_json(summary_path)

    assert summary["rows_evaluated"] == 20000
    assert summary["risk_present_rate"] == 1.0
    assert summary["action_present_rate"] == 1.0
    assert summary["any_driver_mentioned_rate"] == 1.0
    assert summary["disclosure_required_rows"] == 15818
    assert summary["disclosure_present_when_required_rate"] == 1.0
    assert summary["direction_proxy_applicable_rows"] == 20000
    assert 0.32 <= summary["direction_proxy_ok_rate_when_applicable"] <= 0.34
    assert "does not replace human semantic review" in summary["caveat"]


def test_governance_controls_runbook_is_present_and_non_production_claim():
    runbook = read_text(DOCS / "final_governance_controls_runbook.md")

    required = [
        "Candidate Drift Threshold Policy",
        "Thin-File Production Outage Simulation Design",
        "Semantic Validator Proxy",
        "Graph/GNN Evidence-Source Checklist",
        "do not prove production readiness",
    ]

    for phrase in required:
        assert phrase in runbook


def test_final_submission_entrypoint_and_verification_manifest_exist():
    final_submission = read_text(ROOT / "FINAL_SUBMISSION.md")
    assert "Final MSc Thesis Submission Package" in final_submission
    assert "thesis-final-v23-proxy-cohort-diagnostics-20260717" in final_submission
    assert "Operations-summary acceptance" in final_submission
    assert "Audit-complete evidence rendering" in final_submission
    assert "11 passed" in final_submission

    verification_md = read_text(DOCS / "excluded_artifact_verification_manifest.md")
    assert "Excluded Artefact Verification Manifest" in verification_md
    assert "SHA-256" in verification_md
    assert "test_predictions.csv" in verification_md
    assert "narratives_ops_triage_llm_20000_resume_safe.jsonl" in verification_md

    verification_json = read_json(DOCS / "excluded_artifact_verification_manifest.json")
    assert "artifacts" in verification_json
    assert len(verification_json["artifacts"]) >= 10

    paths = {item["path"] for item in verification_json["artifacts"]}
    assert "artifacts/baselines/lgbm_numeric_v1_subsample/test_predictions.csv" in paths
    assert "artifacts/baselines/lgbm_numeric_v1_subsample/validator_policy_sensitivity/validator_policy_sensitivity_summary.json" in paths


def test_validator_policy_figure_and_ci_outputs_exist():
    ci_json = ART / "validator_policy_sensitivity" / "validator_policy_sensitivity_with_ci.json"
    ci_md = ART / "validator_policy_sensitivity" / "validator_policy_sensitivity_with_ci.md"
    fig = ART / "validator_policy_sensitivity" / "validator_policy_sensitivity_figure.svg"
    doc_md = DOCS / "validator_policy_sensitivity_figure.md"

    assert ci_json.exists()
    assert ci_md.exists()
    assert fig.exists()
    assert doc_md.exists()

    result = read_json(ci_json)
    assert result["status"] == "run"
    assert result["ci_method"] == "Wilson score interval, 95%"
    assert len(result["policies"]) == 3

    by_policy = {p["policy"]: p for p in result["policies"]}
    assert by_policy["operations_summary"]["accepted_rows"] == 20000
    assert by_policy["audit_complete_all_driver"]["accepted_rows"] == 7994
    assert by_policy["direction_proxy_confirmed_review"]["accepted_rows"] == 6565


def test_rq2_stability_quantitative_summary_exists_and_supports_interpretation():
    rq2_json = ART / "regeneration_stability" / "rq2_stability_quantitative_summary.json"
    rq2_md = ART / "regeneration_stability" / "rq2_stability_quantitative_summary.md"
    doc_md = DOCS / "rq2_stability_quantitative_summary.md"

    assert rq2_json.exists()
    assert rq2_md.exists()
    assert doc_md.exists()

    result = read_json(rq2_json)
    assert result["status"] == "run"
    assert result["total_regeneration_outputs"] == 800
    assert result["risk_present_all_variants"] is True
    assert result["action_present_all_variants"] is True
    assert result["fallback_zero_all_variants"] is True
    assert result["randomised_variant_driver_overlap"] == 0.763
    assert result["randomised_original_driver_overlap"] == 0.23900000000000002
    assert result["randomised_evidence_following_gap"] > 0.5

    text = read_text(doc_md)
    assert "Randomised evidence-following gap" in text
    assert "automated proxy rather than human semantic validation" in text


def test_proxy_cohort_diagnostics_exist_and_are_bounded():
    proxy_json = ART / "proxy_cohort_diagnostics" / "proxy_cohort_diagnostics_summary.json"
    proxy_md = ART / "proxy_cohort_diagnostics" / "proxy_cohort_diagnostics_summary.md"
    proxy_csv = ART / "proxy_cohort_diagnostics" / "proxy_cohort_diagnostics_by_group.csv"
    doc_md = DOCS / "proxy_cohort_diagnostics.md"

    assert proxy_json.exists()
    assert proxy_md.exists()
    assert proxy_csv.exists()
    assert doc_md.exists()

    result = read_json(proxy_json)
    assert result["status"] == "run"
    assert result["rows"] == 20000
    assert "DeviceType" in result["fields_evaluated"]
    assert "ProductCD" in result["fields_evaluated"]
    assert result["min_group_n"] == 100
    assert result["high_score_threshold_top5pct"] > 0

    text = read_text(doc_md)
    assert "not a fairness audit" in text
    assert "not fairness-performance evidence" in text
    assert "protected-class labels" in text

