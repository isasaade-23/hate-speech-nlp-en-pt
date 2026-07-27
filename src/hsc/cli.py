"""hsc command-line entrypoint. Subcommands are wired to pipeline stages as they land."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="hsc", description="Bilingual hate-speech classifier pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest", help="Extract + normalize the 4 datasets to data/interim")

    h = sub.add_parser("harmonize", help="Build the unified binary corpus (data/processed)")
    h.add_argument("--policy", choices=["strict", "broad"], default="strict")
    h.add_argument("--clean-profile", choices=["light", "heavy"], default="light")
    h.add_argument("--aux", action="store_true", help="Include auxiliary (excluded) sources")
    h.add_argument(
        "--force-gated",
        action="store_true",
        help="Include probe-gated sources even without a recorded probe decision",
    )

    sub.add_parser("probe-dataset2", help="Fase 2: analyze tweets_ip 1/2/3 labels + open gate")

    s = sub.add_parser("split", help="Fase 4: dedup + leakage-safe frozen split")
    s.add_argument("--policy", choices=["strict", "broad"], default="strict")

    lg = sub.add_parser("langid", help="Fase 5: language detection + evaluation")
    lg.add_argument("--policy", choices=["strict", "broad"], default="strict")
    lg.add_argument("--threshold", type=float, default=0.5)

    tr = sub.add_parser("train", help="Fase 7+: train a model from a config")
    tr.add_argument("-c", "--config", required=True, help="path to a model config YAML")
    tr.add_argument("--policy", choices=["strict", "broad"], default=None, help="override config policy")

    ed = sub.add_parser("eda", help="Fase 6: corpus composition figures")
    ed.add_argument("--policy", choices=["strict", "broad"], default="strict")

    sub.add_parser("report", help="Fase 10: build leaderboard + breakdown tables")

    an = sub.add_parser("analyze", help="Fase 9: paired McNemar significance + calibration")
    an.add_argument("--split", default="test", choices=["val", "test"])

    tf = sub.add_parser("transfer", help="Fase 9: cross-domain + cross-lingual transfer")
    tf.add_argument("--policy", choices=["strict", "broad", "both"], default="both")
    tf.add_argument("--seed", type=int, default=42)

    args = parser.parse_args(argv)

    if args.cmd == "ingest":
        from hsc.ingest import run_ingest

        run_ingest()
        return 0

    if args.cmd == "harmonize":
        from hsc.harmonize import build_corpus

        build_corpus(
            policy=args.policy,
            clean_profile=args.clean_profile,
            include_auxiliary=args.aux,
            force_gated=args.force_gated,
        )
        return 0

    if args.cmd == "probe-dataset2":
        from hsc.probe import run

        run()
        return 0

    if args.cmd == "split":
        from hsc.splits import run_split

        run_split(policy=args.policy)
        return 0

    if args.cmd == "langid":
        from hsc.langid import run_langid

        run_langid(policy=args.policy, threshold=args.threshold)
        return 0

    if args.cmd == "train":
        from hsc.train import train_from_config

        train_from_config(args.config, policy_override=args.policy)
        return 0

    if args.cmd == "eda":
        from hsc.eda import run_eda

        run_eda(policy=args.policy)
        return 0

    if args.cmd == "report":
        from hsc.report import build_all

        build_all()
        return 0

    if args.cmd == "analyze":
        from hsc.analysis import run_all

        run_all(split=args.split)
        return 0

    if args.cmd == "transfer":
        from hsc import transfer

        if args.policy == "both":
            transfer.run_all(seed=args.seed)
        else:
            transfer.run_transfer(policy=args.policy, seed=args.seed)
        return 0

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
