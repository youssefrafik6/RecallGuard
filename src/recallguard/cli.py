"""Command-line entry points for data preparation, training and prediction."""

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from recallguard.data import FEATURES, TARGET, load_incidents, load_recalls
from recallguard.inference import IncidentRequest, load_model, predict, save_model
from recallguard.linkage import LinkageConfig, construct_labels, link, review_sample
from recallguard.training import FEATURE_SETS, train


def fingerprint(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(prog="recallguard")
    commands = parser.add_subparsers(required=True, dest="command")
    prepare = commands.add_parser("prepare")
    for field in ("incidents", "recalls", "out"):
        prepare.add_argument(f"--{field}", required=True, type=Path)
    prepare.add_argument("--high", type=float, default=0.82)
    prepare.add_argument("--review", type=float, default=0.57)
    prepare.add_argument("--horizon", type=int, default=365)
    fit = commands.add_parser("train")
    fit.add_argument("--data", type=Path, default=Path("training_data.csv"))
    fit.add_argument("--out", type=Path, default=Path("results/runs"))
    fit.add_argument(
        "--design", choices=["random", "group", "chronological", "all"], default="all"
    )
    fit.add_argument("--feature-set", choices=FEATURE_SETS, default="all")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--smoke", action="store_true")
    for name in ("predict", "model-info"):
        sub = commands.add_parser(name)
        sub.add_argument("--trusted-model", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        incidents, recalls = load_incidents(args.incidents), load_recalls(args.recalls)
        config = LinkageConfig(
            high=args.high, review=args.review, horizon_days=args.horizon
        )
        audit = link(incidents, recalls, config)
        observations = construct_labels(incidents, audit)
        args.out.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.out / "candidates.csv", index=False)
        review_sample(audit, incidents, recalls).to_csv(
            args.out / "review_sample.csv", index=False
        )
        columns = ["report_id", "report_date", *FEATURES, "recall_id", TARGET]
        observations[columns].to_csv(args.out / "observations.csv", index=False)
        summary = {
            "incidents": len(incidents),
            "recalls": len(recalls),
            "candidates": len(audit),
            "linked": int(observations[TARGET].notna().sum()),
            "candidate_confidence_counts": audit["confidence"].value_counts().to_dict(),
            "config": asdict(config),
            "incident_sha256": fingerprint(args.incidents),
            "recall_sha256": fingerprint(args.recalls),
        }
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
    elif args.command == "train":
        frame = load_incidents(args.data, labeled=True)
        data_sha256 = fingerprint(args.data)
        designs = (
            ["random", "group", "chronological"]
            if args.design == "all"
            else [args.design]
        )
        args.out.mkdir(parents=True, exist_ok=True)
        for design in designs:
            bundle = train(frame, design, args.seed, args.feature_set, args.smoke)
            bundle["metadata"].update(
                dataset=args.data.name, dataset_sha256=data_sha256
            )
            save_model(bundle, args.out / f"{design}.joblib")
            (args.out / f"{design}.json").write_text(
                json.dumps(bundle["metadata"], indent=2) + "\n"
            )
    else:
        bundle = load_model(args.trusted_model)
        if args.command == "predict":
            result = predict(
                bundle, IncidentRequest.model_validate_json(sys.stdin.read())
            )
            print(result.model_dump_json(indent=2))
        else:
            print(json.dumps(bundle["metadata"], indent=2))


if __name__ == "__main__":
    main()
