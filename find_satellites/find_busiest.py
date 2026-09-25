"""
find_busiest.py

Scans an existing fov_passes directory (a directory of
<SAT>_results.csv files) and reports either:
    --season    the N busiest observing nights in the directory
    --day DATE  the N busiest moments within one specific night.

A night is labeled by the UTC date it starts on - dusk that day to dawn
the next.

Usage:
    python find_busiest.py FOV_PASSES_DIR --season [--top 10]
    python find_busiest.py FOV_PASSES_DIR --day 2022-11-15 [--top 5]
"""

import argparse
import os
from pathlib import Path

from astropy.time import Time
from astropy import units as u

import find_busy


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "fov_passes_dir",
        help="Directory of <SAT>_results.csv files (find_fov_passes.py's --outdir).",
    )
    window = parser.add_mutually_exclusive_group()
    window.add_argument(
        "--season", action="store_true",
        help="Report the busiest NIGHTS across every night found in the "
             "directory (default if neither --season nor --day is given).",
    )
    window.add_argument(
        "--day", metavar="YYYY-MM-DD",
        help="Restrict to one night (dusk on this date to dawn the next) "
             "and report its busiest MOMENTS instead of busiest nights.",
    )
    parser.add_argument(
        "--top", type=int, default=5,
        help="How many busiest nights/moments to report (default: 5).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output text file path (default: <fov_passes_dir>/"
             "Busiest_nights_top<N>.txt, or Busiest_moments_<DATE>_top<N>.txt "
             "for --day).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    fov_dir = Path(args.fov_passes_dir)

    if args.out:
        out_path = Path(args.out)
    elif args.day:
        out_path = fov_dir / f"Busiest_moments_{args.day}_top{args.top}.txt"
    else:
        out_path = fov_dir / f"Busiest_nights_top{args.top}.txt"

    original_cwd = os.getcwd()
    os.chdir(fov_dir)
    try:
        sat_passes = find_busy.read_sat_passes()
        total_sats = len(sat_passes)

        if args.day:
            sat_pass_keys = {
                sat: {
                    ts for ts, *_ in passes
                    if (Time(ts, scale="utc") - 12 * u.hour).iso.split()[0] == args.day
                }
                for sat, passes in sat_passes.items()
            }
            time_delta_s = find_busy.read_time_delta_s()
            sat_pass_keys = {
                sat: find_busy.collapse_near_duplicates(keys, time_delta_s)
                for sat, keys in sat_pass_keys.items()
            }
        else:
            sat_pass_keys = find_busy.dates_of(sat_passes)

        pass_counts = find_busy.compute_pass_counts(sat_pass_keys)
        if not pass_counts:
            raise SystemExit(
                "No FOV passes found"
                + (f" on the night of {args.day}" if args.day else "")
                + f" in {fov_dir} - nothing to report."
            )
        top_keys = find_busy.top_n_busiest(pass_counts, args.top)

        what = "moments" if args.day else "nights"
        lines = [
            f"Top {len(top_keys)} busiest {what} in {fov_dir}"
            + (f", night of {args.day}" if args.day else ""),
            "=" * 60,
            "",
        ]
        for rank, (key, count) in enumerate(top_keys, start=1):
            sats = find_busy.satellites_at(sat_pass_keys, {key})
            pct = 100 * count / total_sats
            if args.day:
                lines.append(f"{rank}. {key} UTC: {count}/{total_sats} satellites ({pct:.0f}%)")
            else:
                lines.append(
                    f"{rank}. {key} (dusk {key} to dawn next day): "
                    f"{count}/{total_sats} satellites ({pct:.0f}%)"
                )
            lines.append(f"   Satellites: {', '.join(sats)}")
            lines.append("")
    finally:
        os.chdir(original_cwd)

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
