from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.research_report import build_v2_research_report


def main():
    parser = argparse.ArgumentParser(description="Build the Cryptographic Heist v2 research-grade evaluation report.")
    parser.add_argument("--league-manifest", default="logs/v2_checkpoint_league_manifest.json")
    parser.add_argument("--replay-report", default="logs/replay_reports/downtown_chase_seed11_report.json")
    parser.add_argument("--spectator-validation", default="logs/spectator_validation.json")
    parser.add_argument("--operational-readiness", default="logs/operational_readiness.json")
    parser.add_argument("--evidence-verification", default="logs/evidence_bundle_verification.json")
    parser.add_argument("--out", default="logs/v2_research_report.json")
    parser.add_argument("--markdown-out", default="logs/v2_research_report.md")
    parser.add_argument("--min-seeds", type=int, default=5)
    parser.add_argument("--min-steps", type=int, default=1800)
    args = parser.parse_args()

    report = build_v2_research_report(
        league_manifest=args.league_manifest,
        replay_report=args.replay_report,
        spectator_validation=args.spectator_validation,
        operational_readiness=args.operational_readiness,
        evidence_verification=args.evidence_verification,
        out=args.out,
        markdown_out=args.markdown_out,
        thresholds={
            "min_seeds": args.min_seeds,
            "min_steps": args.min_steps,
        },
    )
    scores = report["scores"]
    print(
        f"v2_report passed={str(report['passed']).lower()} "
        f"checks={report['checks_passed']}/{report['checks']} "
        f"overall={scores['league_overall_score']:.3f} "
        f"replay={scores['replay_report_score']:.3f} "
        f"status={report['status']} out={args.out}"
    )


if __name__ == "__main__":
    main()
