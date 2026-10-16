import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from credit_scorecard import cli, pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_end_to_end_small_sample_outputs_exist(mini_run):
    cfg, res = mini_run
    for rel in ("artifacts/scorecard.json", "artifacts/training.json", "artifacts/validation.json",
                "docs/model_development_report.md", "docs/roc.png", "docs/ks.png",
                "docs/calibration.png", "docs/score_distribution.png", "docs/psi.png",
                "docs/cutoff_profit.png"):
        assert (cfg.workdir / rel).stat().st_size > 0, rel
    assert (cfg.workdir / "data" / "synthetic_loans.csv").exists()


def test_end_to_end_results_are_sensible(mini_run):
    _, res = mini_run
    m = res["metrics"]
    assert m["oot"]["auc"] > 0.70 and m["test"]["auc"] > 0.70
    assert 0.2 < m["oot"]["ks"] < 0.7
    assert abs(m["oot"]["gini"] - (2 * m["oot"]["auc"] - 1)) < 1e-9
    assert res["psi"]["score_train_vs_test"] < 0.05  # same population
    assert res["psi"]["score_train_vs_oot"] > res["psi"]["score_train_vs_test"]
    assert res["auc_ci_oot"]["low"] < m["oot"]["auc"] < res["auc_ci_oot"]["high"]
    assert m["oot"]["bad_rate"] > m["train"]["bad_rate"]  # macro shift


def test_challenger_close_to_scorecard_when_sklearn_installed(mini_run):
    pytest.importorskip("sklearn")
    _, res = mini_run
    ch = res["challenger"]
    assert ch is not None and abs(ch["metrics"]["oot"]["auc"] - res["metrics"]["oot"]["auc"]) < 0.05


def test_oracle_is_an_upper_reference(mini_run):
    _, res = mini_run
    assert res["oracle"]["oot_auc_true_pd"] >= res["metrics"]["oot"]["auc"] - 0.02


def test_report_contents(mini_run):
    cfg, _ = mini_run
    text = (cfg.docs / "model_development_report.md").read_text()
    for heading in ("Purpose and scope", "Data quality review", "Methodology", "Validation",
                    "Business layer", "Fairness and governance", "Limitations",
                    "Monitoring plan"):
        assert heading in text
    assert "SYNTHETIC" in text and "IFRS 9" in text and "ILLUSTRATIVE" in text
    assert "TODO" not in text


def test_leakage_column_is_removed_before_modelling(mini_run):
    cfg, _ = mini_run
    tr = json.loads((cfg.artifacts / "training.json").read_text())
    assert any("max_dpd_post_orig" in a for a in tr["quality"]["actions"])
    card = json.loads((cfg.artifacts / "scorecard.json").read_text())
    assert "max_dpd_post_orig" not in card["coefs"] and "age" not in card["coefs"]


def test_split_data_has_no_overlap_and_respects_time(mini_run):
    cfg, _ = mini_run
    prep = pipeline.prepare(cfg)
    ids = [set(d["loan_id"]) for d in (prep.train, prep.test, prep.oot)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    assert sum(len(i) for i in ids) == len(prep.clean)
    assert pd.to_datetime(prep.oot["origination_date"]).min() >= pd.Timestamp(pipeline.OOT_START)
    assert pd.to_datetime(prep.train["origination_date"]).max() < pd.Timestamp(pipeline.OOT_START)
    assert abs(len(prep.test) / (len(prep.test) + len(prep.train)) - 0.30) < 0.01


def test_training_is_deterministic(mini_run):
    cfg, _ = mini_run
    prep = pipeline.prepare(cfg)
    first = (cfg.artifacts / "scorecard.json").read_text()
    pipeline.stage_train(cfg, prep)
    assert (cfg.artifacts / "scorecard.json").read_text() == first


def test_cli_parser_exposes_all_subcommands():
    parser = cli.build_parser()
    for cmd in ("generate", "train", "validate", "score", "report", "run-all"):
        assert parser.parse_args([cmd]).command == cmd


def test_cli_score_with_reason_codes(mini_run, tmp_path):
    cfg, _ = mini_run
    raw = pd.read_csv(cfg.data_path).head(25)
    raw.to_csv(tmp_path / "apps.csv", index=False)
    out = tmp_path / "scored.csv"
    rc = cli.main(["score", "--workdir", str(cfg.workdir), "--input", str(tmp_path / "apps.csv"),
                   "--output", str(out)])
    scored = pd.read_csv(out)
    assert rc == 0 and len(scored) == 25
    assert {"loan_id", "score", "pd", "reason_1", "reason_2", "reason_3"} <= set(scored.columns)
    assert scored["score"].between(300, 900).all()


def test_cli_staged_commands_work_in_sequence(tmp_path, capsys):
    wd = str(tmp_path)
    common = ["--workdir", wd, "--n", "4000", "--seed", "3"]
    assert cli.main(["generate", *common]) == 0
    assert cli.main(["train", *common]) == 0
    assert cli.main(["validate", *common]) == 0
    assert cli.main(["report", *common]) == 0
    assert "SYNTHETIC" in capsys.readouterr().out
    assert (tmp_path / "docs" / "model_development_report.md").exists()


def test_cli_reports_missing_data_cleanly(tmp_path, capsys):
    assert cli.main(["train", "--workdir", str(tmp_path)]) == 2
    assert "generate" in capsys.readouterr().err


def test_shipped_dataset_is_small_enough_and_labelled_synthetic():
    csv = ROOT / "data" / "synthetic_loans.csv"
    if not csv.exists():
        pytest.skip("dataset not shipped")
    assert csv.stat().st_size < 15 * 1024 * 1024
    meta = json.loads((ROOT / "data" / "synthetic_loans.meta.json").read_text())
    assert meta["synthetic"] is True


def test_no_em_or_en_dashes_anywhere_in_the_repository():
    bad = []
    for path in ROOT.rglob("*"):
        if path.suffix in {".py", ".md", ".toml", ".yml", ".json", ".txt", ".csv"} \
                and ".git" not in path.parts and "egg-info" not in str(path) \
                and ".pytest_cache" not in path.parts and ".ruff_cache" not in path.parts:
            text = path.read_text(errors="ignore")
            if chr(0x2014) in text or chr(0x2013) in text:
                bad.append(str(path.relative_to(ROOT)))
    assert not bad, bad


def test_scorecard_scores_are_in_a_plausible_range(mini_run):
    cfg, _ = mini_run
    from credit_scorecard.scorecard import Scorecard

    card = Scorecard.load(cfg.artifacts / "scorecard.json")
    s = card.score(pipeline.prepare(cfg).oot)
    assert np.percentile(s, 1) > 300 and np.percentile(s, 99) < 800
