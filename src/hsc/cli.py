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
    # Future: split, langid, train, evaluate, report

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

    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
