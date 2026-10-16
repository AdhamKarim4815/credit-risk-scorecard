"""Command line interface: ``credit-scorecard {generate,train,validate,score,report,run-all}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from credit_scorecard import __version__, generate, pipeline


def _config(args: argparse.Namespace) -> pipeline.Config:
    return pipeline.Config(workdir=Path(args.workdir), n=args.n, seed=args.seed)


def _cmd_generate(args) -> int:
    cfg = _config(args)
    df = pipeline.stage_generate(cfg)
    print(f"wrote {len(df):,} SYNTHETIC loans to {cfg.data_path}")
    return 0


def _cmd_train(args) -> int:
    cfg = _config(args)
    sc, info = pipeline.stage_train(cfg)
    print(f"trained scorecard on {info['data']['splits']['train']['n']:,} loans; "
          f"characteristics: {', '.join(sc.features)}")
    print(f"saved {cfg.artifacts / 'scorecard.json'}")
    return 0


def _cmd_validate(args) -> int:
    cfg = _config(args)
    res = pipeline.stage_validate(cfg)
    for name, m in res["metrics"].items():
        print(f"{name:>5}: AUC {m['auc']:.3f}  Gini {m['gini']:.3f}  KS {m['ks']:.3f}  "
              f"bad rate {m['bad_rate']:.2%}")
    print(f"score PSI (train vs out-of-time): {res['psi']['score_train_vs_oot']:.3f}")
    print(f"charts written to {cfg.docs}")
    return 0


def _cmd_score(args) -> int:
    from credit_scorecard.scorecard import Scorecard

    cfg = _config(args)
    sc = Scorecard.load(cfg.artifacts / "scorecard.json")
    if args.input:
        df = pipeline.add_features(pd.read_csv(args.input))
    else:  # demo: five applicants from the synthetic file
        df = pipeline.add_features(pd.read_csv(cfg.data_path).sample(5, random_state=1))
    out = sc.score_frame(df)
    if "loan_id" in df:
        out.insert(0, "loan_id", df["loan_id"])
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"scored {len(out):,} applicants -> {args.output}")
    else:
        with pd.option_context("display.width", 200, "display.max_colwidth", 60):
            print(out.to_string(index=False))
    return 0


def _cmd_report(args) -> int:
    path = pipeline.stage_report(_config(args))
    print(f"wrote {path}")
    return 0


def _cmd_run_all(args) -> int:
    pipeline.run_all(_config(args))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="credit-scorecard",
        description="Retail credit risk (PD) scorecard pipeline. All data is SYNTHETIC.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workdir", default=".", help="project root holding data/, artifacts/, "
                                                      "docs/ (default: current directory)")
    common.add_argument("--n", type=int, default=generate.N_DEFAULT, help="loans to generate")
    common.add_argument("--seed", type=int, default=generate.SEED, help="random seed")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", parents=[common], help="generate the synthetic dataset"
                   ).set_defaults(func=_cmd_generate)
    sub.add_parser("train", parents=[common], help="data quality, binning, fit, scorecard"
                   ).set_defaults(func=_cmd_train)
    sub.add_parser("validate", parents=[common], help="validation metrics, charts, business layer"
                   ).set_defaults(func=_cmd_validate)
    sc = sub.add_parser("score", parents=[common], help="score applicants with reason codes")
    sc.add_argument("--input", help="CSV of applicants (default: 5 demo rows)")
    sc.add_argument("--output", help="write scored CSV here instead of printing")
    sc.set_defaults(func=_cmd_score)
    sub.add_parser("report", parents=[common], help="write the Markdown model development report"
                   ).set_defaults(func=_cmd_report)
    sub.add_parser("run-all", parents=[common], help="run every stage end to end"
                   ).set_defaults(func=_cmd_run_all)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
