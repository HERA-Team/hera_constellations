"""
run_pipeline.py

End-to-end pipeline for a single constellation on a single night at HERA:

    1. Fetch TLEs (fetch_constellation_tles.py) - the archival TLE closest
       to --date for every satellite in --constellation.
    2. Find every FOV crossing that night (find_fov_passes.py) - propagates
       every satellite and records each night-time timestamp it's inside
       HERA's beam.
    3. Find the busiest moment (find_busy.py) - the exact timestamp with
       the most satellites simultaneously in the beam, from step 2's
       per-satellite results.
    4. Plot that busiest window (beam_vs_starlink.py) - RA/Dec trajectories
       of every satellite in the beam during a --window-minutes window
       centred on the busiest moment from step 3.

All four scripts (plus this one) live together in find_satellites/.

Note step order: this runs find_fov_passes (step 2 above) *before* the
busiest-window plot, even though "find crossings" reads like a final
step - the plot needs to know the busiest moment, and that can only come
from find_fov_passes' per-satellite results. Both deliverables (the full
set of FOV crossings, and the busiest-window plot) fall out of the same
run; there's no separate "step 4" propagation.

Requires a free Space-Track.org account for step 1 - see
fetch_constellation_tles.py's docstring. Credentials are read from
SPACETRACK_USER/SPACETRACK_PASS env vars, or prompted for interactively.

Usage:
    python run_pipeline.py --date 2022-11-13 --constellation Starlink
    python run_pipeline.py --date 2022-11-13 --constellation Orbcomm \
        --window-minutes 15 --outdir my_runs

Outputs, all under --outdir/<constellation>_<date>/ (default outdir:
Pipeline_runs/ next to this script):
    tle/<date>_<constellation>_tles.txt   - fetched TLEs (step 1)
    fov_passes/<SAT>_results.csv          - per-satellite FOV crossings (step 2)
    fov_passes/Running_details.txt        - run parameters/resource usage (step 2)
    fov_passes/Time-satellite_correspondence.txt,
    fov_passes/Busiest_time_positions.txt - find_busy.py's own reference
                                             outputs (step 3, time-of-day
                                             aggregated across the CSVs -
                                             for one night this is one date,
                                             so equivalent to this run)
    beam_busiest_window.png               - trajectory plot (step 4)
    pipeline_summary.txt                  - busiest moment found, satellite
                                             count there, and paths to all
                                             of the above
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from astropy.time import Time, TimeDelta
from astropy import units as u

SCRIPT_DIR = Path(__file__).resolve().parent
FETCH_TLES = SCRIPT_DIR / "fetch_constellation_tles.py"
FIND_FOV_PASSES = SCRIPT_DIR / "find_fov_passes.py"
BEAM_VS_STARLINK = SCRIPT_DIR / "beam_vs_starlink.py"

# find_busy.py now lives alongside this script (find_satellites/), so a
# plain import works - Python puts a script's own directory on sys.path.
import find_busy

# Matches the "Time considered (night-time only, ...)" line find_fov_passes.py
# writes to Running_details.txt, so the actual dusk-to-dawn window (Sun below
# HERA's horizon, per astropy's get_sun) can be reported here too, without
# recomputing it - see read_night_window() below.
NIGHT_WINDOW_RE = re.compile(
    r"Time considered \(night-time only.*?\): from (?P<start>.+?) to (?P<end>.+)$",
    re.MULTILINE,
)


def constellation_type(value):
    """Normalize e.g. 'starlink'/'STARLINK' -> 'Starlink'; reject anything else."""
    normalized = value.strip().capitalize()
    if normalized not in ("Starlink", "Orbcomm"):
        raise argparse.ArgumentTypeError(
            f"must be 'Starlink' or 'Orbcomm' (case-insensitive), got {value!r}"
        )
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date", required=True,
        help="Date of interest, e.g. 2022-11-13. Both the TLE fetch and the "
             "FOV-pass search center on the night starting on this date "
             "(dusk this day to dawn the next).",
    )
    parser.add_argument(
        "--constellation", required=True, type=constellation_type,
        help="'Starlink' or 'Orbcomm' (case-insensitive).",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=10.0,
        help="Width, in minutes, of the trajectory-plot window centred on "
             "the busiest moment found (default: 10).",
    )
    parser.add_argument(
        "--busy-threshold", type=float, default=0.90,
        help="Passed through to find_busy.py's --threshold, used only for "
             "its own reference output files (default: 0.90).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap the number of satellites propagated in the busiest-window "
             "plot (step 4 only; step 2 always covers every satellite in "
             "the fetched TLE file). Default: all satellites.",
    )
    parser.add_argument(
        "--outdir", default=str(SCRIPT_DIR / "Pipeline_runs"),
        help="Base directory for run outputs; a <constellation>_<date> "
             "subdirectory is created under it (default: .)",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bars in steps 2 and 4 (passed "
             "through to find_fov_passes.py and beam_vs_starlink.py). "
             "Useful for Slurm/batch jobs, where in-place redraws just "
             "clutter the log file.",
    )
    return parser.parse_args()


def run_step(description, cmd):
    print(f"\n=== {description} ===")
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def read_night_window(running_details_path):
    """Pull the actual dusk-to-dawn bounds out of find_fov_passes.py's
    Running_details.txt (written by step 2), rather than re-deriving them
    with a second get_sun/AltAz computation here."""
    match = NIGHT_WINDOW_RE.search(running_details_path.read_text())
    if not match:
        return None, None
    return match.group("start"), match.group("end")


def main():
    args = parse_args()

    # Validate --date early, before doing any work.
    Time(args.date, scale="utc")

    run_dir = Path(args.outdir) / f"{args.constellation.lower()}_{args.date}"
    tle_dir = run_dir / "tle"
    fov_dir = run_dir / "fov_passes"
    tle_dir.mkdir(parents=True, exist_ok=True)
    fov_dir.mkdir(parents=True, exist_ok=True)

    tle_file = tle_dir / f"{args.date}_{args.constellation.lower()}_tles.txt"

    # --- Step 1: fetch TLEs -------------------------------------------------
    if os.path.exists(tle_file):
        print("Step 1/4: fetching TLEs from Space-Track")
        print(f"{tle_file} already exists! reading from it...")
    else:
        run_step(
            "Step 1/4: fetching TLEs from Space-Track",
            [
                sys.executable, str(FETCH_TLES),
                "--constellation", args.constellation.upper(),
                "--date", args.date,
                "--window-days", '3.0',
                "--out", str(tle_file),
            ],
        )

    # --- Step 2: find every FOV crossing that night -------------------------
    fov_cmd = [
        sys.executable, str(FIND_FOV_PASSES),
        str(tle_file),
        "--day", args.date,
        "--outdir", str(fov_dir),
    ]
    if args.no_progress:
        fov_cmd.append("--no-progress")
    run_step(
        "Step 2/4: finding FOV crossings for the night of "
        f"{args.date} (needed before the busiest moment can be found)",
        fov_cmd,
    )

    night_start, night_end = read_night_window(fov_dir / "Running_details.txt")
    if night_start:
        print(f"Night considered (Sun below horizon at HERA): "
              f"{night_start} to {night_end} UTC")

    # --- Step 3: find the busiest moment -------------------------------------
    print("\n=== Step 3/4: finding the busiest moment ===")

    original_cwd = os.getcwd()
    os.chdir(fov_dir)
    try:
        sat_passes = find_busy.read_sat_passes()
        total_sats = len(sat_passes)

        # find_busy.py's own "time" mode aggregates time-of-day *across*
        # potentially many nights, which isn't quite what we want here: for
        # a single night, counting by the *full* timestamp (not just
        # HH:MM:SS) pins down one exact, unambiguous moment instead. Reuses
        # find_busy.py's generic compute_pass_counts() over full timestamps.
        sat_pass_timestamps = {
            sat: {ts for ts, *_ in passes} for sat, passes in sat_passes.items()
        }
        pass_counts = find_busy.compute_pass_counts(sat_pass_timestamps)
        if not pass_counts:
            raise SystemExit(
                f"No FOV passes found for any {args.constellation} satellite "
                f"on the night of {args.date} - nothing to plot."
            )
        busiest_count = max(pass_counts.values())
        busiest_timestamps = sorted(
            ts for ts, count in pass_counts.items() if count == busiest_count
        )
        busiest_ts = busiest_timestamps[0]  # chronologically first if tied

        # Also write find_busy.py's own reference files (busy/sparse
        # time-of-day breakdown etc.) into fov_dir, for cross-checking.
        find_busy.run_time_mode(sat_passes, args.busy_threshold)
    finally:
        os.chdir(original_cwd)

    print(
        f"Busiest moment: {busiest_ts} UTC "
        f"({busiest_count}/{total_sats} satellites in beam"
        + (f", tied with {len(busiest_timestamps) - 1} other moment(s)"
           if len(busiest_timestamps) > 1 else "")
        + ")"
    )

    # --- Step 4: plot the busiest window -------------------------------------
    anchor = Time(busiest_ts, scale="utc")
    half_window = TimeDelta(args.window_minutes / 2 * u.minute)
    beam_start = (anchor - half_window).isot
    beam_end = (anchor + half_window).isot

    beam_plot = run_dir / "beam_busiest_window.png"
    beam_cmd = [
        sys.executable, str(BEAM_VS_STARLINK),
        str(tle_file),
        "--start", beam_start,
        "--end", beam_end,
        "--constellation", args.constellation,
        "--outfile", str(beam_plot),
        # Step 2 already found which satellites pass through the beam when -
        # reuse that instead of re-propagating the entire TLE file just to
        # find the same handful again.
        "--from-passes-dir", str(fov_dir),
    ]
    if args.limit is not None:
        beam_cmd += ["--limit", str(args.limit)]
    if args.no_progress:
        beam_cmd.append("--no-progress")
    run_step(
        f"Step 4/4: plotting the {args.window_minutes:.0f} min window "
        f"around the busiest moment",
        beam_cmd,
    )

    # --- Summary --------------------------------------------------------------
    summary_path = run_dir / "pipeline_summary.txt"
    night_line = (f"{night_start} to {night_end} UTC" if night_start else "unknown")
    summary = (
        f"Pipeline run: {args.constellation}, night of {args.date}\n"
        f"{'=' * 60}\n\n"
        f"Night considered:    {night_line} "
        f"(Sun below horizon at HERA)\n"
        f"TLE file:            {tle_file}\n"
        f"FOV-pass CSVs:       {fov_dir}/*_results.csv\n"
        f"Busiest moment:      {busiest_ts} UTC "
        f"({busiest_count}/{total_sats} satellites in beam)\n"
        f"Plot window:         {beam_start} to {beam_end} UTC "
        f"({args.window_minutes:.0f} min)\n"
        f"Trajectory plot:     {beam_plot}\n"
    )
    summary_path.write_text(summary)
    print(f"\n{summary}\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
