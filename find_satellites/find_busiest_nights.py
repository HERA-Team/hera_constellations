"""
find_busiest_nights.py

Scans an existing fov_passes directory (produced by find_fov_passes.py, or
by run_pipeline.py's fov_passes/ subdirectory - a directory of
<SAT>_results.csv files) and reports the N busiest observing nights
(default 5): the nights with the most satellites simultaneously crossing
HERA's FOV at some point.

A night is labeled by the UTC date it starts on - dusk that day to dawn
the next - using find_busy.dates_of()'s night-aware grouping, the same one
run_pipeline.py's --season mode uses to find the single busiest night, so
results here are directly consistent with a run_pipeline.py --season run
over the same fov_passes directory.

Usage:
    python find_busiest_nights.py FOV_PASSES_DIR
    python find_busiest_nights.py FOV_PASSES_DIR --top 10 --out top10.txt
"""

import argparse
import os
from pathlib import Path

import find_busy


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "fov_passes_dir",
        help="Directory of <SAT>_results.csv files (find_fov_passes.py's --outdir).",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="How many busiest nights to report (default: 5).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output text file path (default: <fov_passes_dir>/Busiest_nights_top<N>.txt).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fov_dir = Path(args.fov_passes_dir)
    out_path = Path(args.out) if args.out else fov_dir / f"Busiest_nights_top{args.top}.txt"

    original_cwd = os.getcwd()
    os.chdir(fov_dir)
    try:
        sat_passes = find_busy.read_sat_passes()
        total_sats = len(sat_passes)
        sat_pass_nights = find_busy.dates_of(sat_passes)
        pass_counts = find_busy.compute_pass_counts(sat_pass_nights)
        if not pass_counts:
            raise SystemExit(f"No FOV passes found in {fov_dir} - nothing to report.")
        top_nights = find_busy.top_n_busiest(pass_counts, args.top)

        lines = [
            f"Top {len(top_nights)} busiest observing nights in {fov_dir}",
            "=" * 60,
            "",
        ]
        for rank, (night, count) in enumerate(top_nights, start=1):
            sats_that_night = find_busy.satellites_at(sat_pass_nights, {night})
            pct = 100 * count / total_sats
            lines.append(
                f"{rank}. {night} (dusk {night} to dawn next day): "
                f"{count}/{total_sats} satellites ({pct:.0f}%)"
            )
            lines.append(f"   Satellites: {', '.join(sats_that_night)}")
            lines.append("")
    finally:
        os.chdir(original_cwd)

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
