"""Fetch SUPPORT2 from UCI using ucimlrepo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Download the UCI SUPPORT2 dataset."
    )

    parser.add_argument(
        "--output",
        default="data/raw/support2.csv",
        help="Destination CSV path.",
    )

    return parser.parse_args()


def main() -> None:
    """Download and save the complete SUPPORT2 table."""

    args = parse_arguments()

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit(
            "ucimlrepo is not installed. Run: "
            "pip install -e '.[dev,data]'"
        ) from exc

    print("Fetching SUPPORT2 from UCI repository...")

    dataset = fetch_ucirepo(
        id=880
    )

    original = getattr(
        dataset.data,
        "original",
        None,
    )

    if (
        isinstance(original, pd.DataFrame)
        and not original.empty
    ):
        frame = original.copy()
    else:
        parts = []

        features = getattr(
            dataset.data,
            "features",
            None,
        )

        targets = getattr(
            dataset.data,
            "targets",
            None,
        )

        if isinstance(features, pd.DataFrame):
            parts.append(features)

        if isinstance(targets, pd.DataFrame):
            parts.append(targets)

        if not parts:
            raise RuntimeError(
                "The UCI client returned no usable "
                "SUPPORT2 DataFrame."
            )

        frame = pd.concat(
            parts,
            axis=1,
        )

        # Remove duplicate columns if a field was returned in both sections.
        frame = frame.loc[
            :,
            ~frame.columns.duplicated(),
        ]

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        output,
        index=False,
    )

    print(
        f"Saved {len(frame):,} rows and "
        f"{frame.shape[1]} columns to {output}"
    )

    print("\nFirst 10 columns:")
    print(
        list(frame.columns[:10])
    )


if __name__ == "__main__":
    main()
