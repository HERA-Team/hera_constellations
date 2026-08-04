"""
run_pipeline.py

End-to-end pipeline for a single constellation, over either a single night
or a full observing season at HERA - --day and --season are mutually
exclusive:

    1. Fetch TLEs (fetch_constellation_tles.py) - one archival TLE snapshot
       per satellite in --constellation, closest to an anchor date: --day
       itself for --day, or Jan 1 of the following year for --season
       (splits the difference across the Sept-Apr season).
    2. Find every FOV crossing (find_fov_passes.py --day/--season) -
       propagates every satellite and records each night-time timestamp
       it's inside HERA's beam, over the one night or the whole season.
    3. Find the busiest moment, --day (find_busy.py) - the exact timestamp
       with the most satellites simultaneously in the beam - OR the busiest
       NIGHT, --season - the single night (across the whole season) with
       the most satellites passing through the beam at some point. A
       "night" is always labeled by the UTC date it *starts* on: "busiest
       night: 2022-11-23" means the interval from dusk on 2022-11-23 to
       dawn on 2022-11-24 (find_busy.dates_of() defines nights this way
       specifically so this never gets split across the UTC-midnight
       boundary that every HERA night straddles).
    4. Plot that busiest window (beam_vs_starlink.py) - RA/Dec trajectories
       of every satellite in the beam during a --window-minutes window
       centred on the busiest moment (--day), or across the *entire*
       busiest night, dusk to dawn (--season).

All four scripts (plus this one) live together in find_satellites/.

Note step order: this runs find_fov_passes (step 2 above) *before* the
busiest-window plot, even though "find crossings" reads like a final
step - the plot needs to know the busiest moment/night, and that can only
come from find_fov_passes' per-satellite results. Both deliverables (the
full set of FOV crossings, and the busiest-window plot) fall out of the
same run; there's no separate "step 4" propagation.

Requires a free Space-Track.org account for step 1 - see
fetch_constellation_tles.py's docstring. Credentials are read from
SPACETRACK_USER/SPACETRACK_PASS env vars, or prompted for interactively.

Usage:
    python run_pipeline.py --day 2022-11-13 --constellation Starlink
    python run_pipeline.py --day 2022-11-13 --constellation Orbcomm \
        --window-minutes 15 --outdir my_runs
    python run_pipeline.py --season 2022 --constellation Starlink

Outputs, all under --outdir/<constellation>_<label>/ (default outdir:
Pipeline_runs/ next to this script; <label> is the --day date, or
"season_<YEAR>" for --season):
    tle/<label>_<constellation>_tles.txt   - fetched TLEs (step 1)
    fov_passes/<SAT>_results.csv           - per-satellite FOV crossings (step 2)
    fov_passes/Running_details.txt         - run parameters/resource usage (step 2)
    fov_passes/Time-satellite_correspondence.txt,
    fov_passes/Busiest_time_positions.txt  - find_busy.py's own reference
                                              outputs, --day only (time-of-day
                                              aggregated across the CSVs -
                                              for one night this is one date,
                                              so equivalent to this run)
    fov_passes/Dates_with_missing_pass.txt,
    fov_passes/Date-satellite_missing.txt,
    fov_passes/Busiest_day_positions.txt   - find_busy.py's own reference
                                              outputs, --season only (busiest-
                                              night breakdown; "Busiest_day_
                                              positions.txt" is find_busy.py's
                                              own filename, but its counting
                                              is the same night-aware one
                                              used for busiest_key above)
    beam_busiest_window_<date>.png         - trajectory plot (step 4); <date>
                                              is --day, or the busiest night
                                              found for --season
    top<N>_busiest_moments.txt (--day) or
    top<N>_busiest_nights.txt (--season)   - the --top-n busiest moments/
                                              nights (default 5, beyond just
                                              the single one the plot is
                                              centred on), each with its
                                              satellite count and which
                                              satellites were present
    pipeline_summary.txt                   - busiest moment/night found,
                                              satellite count there, and
                                              paths to all of the above

find_busiest.py separately reports the --top N busiest nights (--season)
or busiest moments within one night (--day DATE) from an existing
fov_passes/ directory, without re-running the pipeline:
    python find_busiest.py path/to/fov_passes --season --top 10
    python find_busiest.py path/to/fov_passes --day 2022-11-15 --top 5
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
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument(
        "--day", metavar="YYYY-MM-DD",
        help="Run the pipeline for a single night: the archival TLE fetch "
             "and FOV-pass search center on the night starting on this "
             "date (dusk this day to dawn the next). The busiest MOMENT "
             "that night is found, and a --window-minutes plot centred on "
             "it is produced.",
    )
    window.add_argument(
        "--season", type=int, metavar="YEAR",
        help="Run the pipeline for a full observing season, Sept 1 of "
             "YEAR to Apr 30 of YEAR+1 (e.g. --season 2022 for "
             "2022-09-01 - 2023-04-30). TLEs are fetched anchored on Jan 1 "
             "of YEAR+1. Instead of a busiest moment, the busiest single "
             "NIGHT across the whole season is found and plotted in full "
             "(dusk to dawn) - --window-minutes is ignored.",
    )
    parser.add_argument(
        "--constellation", required=True, type=constellation_type,
        help="'Starlink' or 'Orbcomm' (case-insensitive).",
    )
    parser.add_argument(
        "--window-minutes", type=float, default=10.0,
        help="Width, in minutes, of the trajectory-plot window centred on "
             "the busiest moment found (default: 10). --day mode only; "
             "--season always plots the entire busiest night.",
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
        "--top-n", type=int, default=5,
        help="How many busiest moments (--day) or busiest nights (--season) "
             "to report, beyond the single busiest one the plot is centred "
             "on (default: 5). Written to top<N>_busiest_moments.txt or "
             "top<N>_busiest_nights.txt in the run directory.",
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

    # Validate --day/--season early, before doing any work. The --season
    # construction mirrors find_fov_passes.py's own t0 for --season
    # (find_fov_passes.py's Sept-1 Time(...) call), so if this passes,
    # find_fov_passes.py's own subprocess call won't choke on the same one.
    if args.day:
        Time(args.day, scale="utc")
    else:
        Time(f"{args.season}-09-01T00:00:00", format="isot", scale="utc")

    # label names run_dir/tle_file: the date itself for --day, or
    # "season_YEAR" for --season (there's no single date to use there).
    label = args.day if args.day else f"season_{args.season}"

    run_dir = Path(args.outdir) / f"{args.constellation.lower()}_{label}"
    tle_dir = run_dir / "tle"
    fov_dir = run_dir / "fov_passes"
    tle_dir.mkdir(parents=True, exist_ok=True)
    fov_dir.mkdir(parents=True, exist_ok=True)

    tle_file = tle_dir / f"{label}_{args.constellation.lower()}_tles.txt"

    # --- Step 1: fetch TLEs -------------------------------------------------
    # A single TLE snapshot is fetched regardless of mode - fetch_constellation_tles.py
    # always keeps just the archival TLE closest to one anchor date. For
    # --season, that anchor is Jan 1 of the following year (splits the
    # difference across the Sept-Apr season, rather than anchoring at either
    # end and leaving the far end of the season badly stale).
    if args.day:
        fetch_date = args.day
    else:
        fetch_date = f"{args.season + 1}-01-01"
        print(f"--season {args.season}: TLEs will be fetched anchored on {fetch_date}.")

    if os.path.exists(tle_file):
        print("Step 1/4: fetching TLEs from Space-Track")
        print(f"{tle_file} already exists! reading from it...")
    else:
        run_step(
            "Step 1/4: fetching TLEs from Space-Track",
            [
                sys.executable, str(FETCH_TLES),
                "--constellation", args.constellation.upper(),
                "--date", fetch_date,
                "--window-days", '3.0',
                "--out", str(tle_file),
            ],
        )

    # --- Step 2: find every FOV crossing (one night, or the whole season) --
    fov_cmd = [
        sys.executable, str(FIND_FOV_PASSES),
        str(tle_file),
        "--outdir", str(fov_dir),
    ]
    fov_cmd += ["--day", args.day] if args.day else ["--season", str(args.season)]
    if args.no_progress:
        fov_cmd.append("--no-progress")
    if args.day:
        step2_desc = ("Step 2/4: finding FOV crossings for the night of "
                      f"{args.day} (needed before the busiest moment can be found)")
    else:
        step2_desc = ("Step 2/4: finding FOV crossings across the "
                      f"{args.season}-{args.season + 1} season (needed before "
                      "the busiest night can be found)")
    run_step(step2_desc, fov_cmd)

    night_start, night_end = read_night_window(fov_dir / "Running_details.txt")
    if night_start:
        span_label = "Night considered" if args.day else "Season considered"
        print(f"{span_label}:  {night_start} to {night_end} UTC")

    # --- Step 3: find the busiest moment (--day) or busiest night (--season) --
    print(f"\n=== Step 3/4: finding the busiest {'moment' if args.day else 'night'} ===")

    original_cwd = os.getcwd()
    os.chdir(fov_dir)
    try:
        sat_passes = find_busy.read_sat_passes()
        total_sats = len(sat_passes)

        if args.day:
            # find_busy.py's own "time" mode aggregates time-of-day *across*
            # potentially many nights, which isn't quite what we want here:
            # for a single night, counting by the *full* timestamp (not
            # just HH:MM:SS) pins down one exact, unambiguous moment
            # instead. Reuses find_busy.py's generic compute_pass_counts()
            # over full timestamps.
            sat_pass_keys = {
                sat: {ts for ts, *_ in passes} for sat, passes in sat_passes.items()
            }
            # Collapse each satellite's own near-duplicate timestamps (the
            # same continuous crossing sampled at several consecutive
            # time_delta steps, e.g. 17:13:25/30/35) into one representative
            # timestamp per crossing - otherwise one pass inflates
            # pass_counts/the top-n list with several entries for what's
            # really a single event.
            time_delta_s = find_busy.read_time_delta_s()
            sat_pass_keys = {
                sat: find_busy.collapse_near_duplicates(keys, time_delta_s)
                for sat, keys in sat_pass_keys.items()
            }
            pass_counts = find_busy.compute_pass_counts(sat_pass_keys)
            empty_msg = (f"No FOV passes found for any {args.constellation} "
                         f"satellite on the night of {args.day} - nothing to plot.")
        else:
            # Busiest NIGHT across the season, via find_busy.dates_of() -
            # which labels a night by the UTC date it starts on (dusk to
            # dawn), so this and find_busy.run_day_mode()'s own reference
            # files below always agree on which night is busiest.
            sat_pass_keys = find_busy.dates_of(sat_passes)
            pass_counts = find_busy.compute_pass_counts(sat_pass_keys)
            empty_msg = (f"No FOV passes found for any {args.constellation} "
                         f"satellite in the {args.season}-{args.season + 1} "
                         "season - nothing to plot.")

        if not pass_counts:
            raise SystemExit(empty_msg)

        busiest_count = max(pass_counts.values())
        busiest_keys = sorted(
            key for key, count in pass_counts.items() if count == busiest_count
        )
        busiest_key = busiest_keys[0]  # chronologically/lexicographically first if tied

        # Top --top-n list (beyond just the single busiest_key the plot is
        # centred on) - satellites_at() reuses the same sat_pass_keys built
        # above, so this is consistent with busiest_key regardless of mode.
        top_n_list = [
            (key, count, find_busy.satellites_at(sat_pass_keys, {key}))
            for key, count in find_busy.top_n_busiest(pass_counts, args.top_n)
        ]

        # Also write find_busy.py's own reference files, for cross-checking.
        if args.day:
            find_busy.run_time_mode(sat_passes, args.busy_threshold)
        else:
            find_busy.run_day_mode(sat_passes, args.busy_threshold)
    finally:
        os.chdir(original_cwd)

    if args.day:
        print(
            f"Busiest moment: {busiest_key} UTC "
            f"({busiest_count}/{total_sats} satellites in beam"
            + (f", tied with {len(busiest_keys) - 1} other moment(s)"
               if len(busiest_keys) > 1 else "")
            + ")"
        )
    else:
        night_after = (Time(busiest_key, scale="utc") + 1 * u.day).strftime("%Y-%m-%d")
        print(
            f"Busiest night: {busiest_key} "
            f"(dusk {busiest_key} to dawn {night_after}) "
            f"({busiest_count}/{total_sats} satellites in beam"
            + (f", tied with {len(busiest_keys) - 1} other night(s)"
               if len(busiest_keys) > 1 else "")
            + ")"
        )

    what = "moments" if args.day else "nights"
    top_n_path = run_dir / f"top{args.top_n}_busiest_{what}.txt"
    top_n_lines = [
        f"Top {len(top_n_list)} busiest {what}, {args.constellation}"
        + (f", night of {args.day}" if args.day
           else f", season {args.season}-{args.season + 1}"),
        "=" * 60,
        "",
    ]
    for rank, (key, count, sats) in enumerate(top_n_list, start=1):
        if args.day:
            top_n_lines.append(f"{rank}. {key} UTC: {count}/{total_sats} satellites")
        else:
            key_after = (Time(key, scale="utc") + 1 * u.day).strftime("%Y-%m-%d")
            top_n_lines.append(
                f"{rank}. {key} (dusk {key} to dawn {key_after}): "
                f"{count}/{total_sats} satellites"
            )
        top_n_lines.append(f"   Satellites: {', '.join(sats)}")
        top_n_lines.append("")
    top_n_path.write_text("\n".join(top_n_lines))
    print(f"Top {args.top_n} busiest {what} saved to {top_n_path}")

    # --- Step 4: plot the busiest window (--day) or entire busiest night (--season) --
    if args.day:
        anchor = Time(busiest_key, scale="utc")
        half_window = TimeDelta(args.window_minutes / 2 * u.minute)
        beam_start = (anchor - half_window).isot
        beam_end = (anchor + half_window).isot
        window_desc = f"the {args.window_minutes:.0f} min window around the busiest moment"
        plot_date_label = args.day
    else:
        # Padded noon-to-noon window for this specific night, same
        # convention find_fov_passes.py's --day uses - beam_vs_starlink.py's
        # own internal filter_to_night() (get_sun-based) restricts what's
        # actually plotted/propagated to real night-time samples, and its
        # plot title now reports the true (night-filtered) span rather than
        # this padded request window.
        day_start = Time(busiest_key, scale="utc")
        beam_start = (day_start + 12 * u.hour).isot
        beam_end = (day_start + 36 * u.hour).isot
        window_desc = f"the entire busiest night ({busiest_key}, dusk to dawn)"
        plot_date_label = busiest_key

    beam_plot = run_dir / f"beam_busiest_window_{plot_date_label}.png"
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
    run_step(f"Step 4/4: plotting {window_desc}", beam_cmd)

    # --- Summary --------------------------------------------------------------
    summary_path = run_dir / "pipeline_summary.txt"
    night_line = (f"{night_start} to {night_end} UTC" if night_start else "unknown")

    if args.day:
        header = f"Pipeline run: {args.constellation}, night of {args.day}"
        span_label = "Night considered"
        busiest_line = (f"Busiest moment:      {busiest_key} UTC "
                         f"({busiest_count}/{total_sats} satellites in beam)\n")
        plot_line = (f"Plot window:         {beam_start} to {beam_end} UTC "
                     f"({args.window_minutes:.0f} min)\n")
    else:
        header = f"Pipeline run: {args.constellation}, season {args.season}-{args.season + 1}"
        span_label = "Season considered"
        busiest_line = (f"Busiest night:       {busiest_key} "
                         f"({busiest_count}/{total_sats} satellites in beam)\n")
        plot_line = "Plot window:         entire busiest night, dusk to dawn " \
                    "(see plot title for exact bounds)\n"

    summary = (
        f"{header}\n"
        f"{'=' * 60}\n\n"
        f"{span_label}:{' ' * max(1, 20 - len(span_label) - 1)}{night_line} "
        f"(Sun below horizon at HERA)\n"
        f"TLE file:            {tle_file}\n"
        f"FOV-pass CSVs:       {fov_dir}/*_results.csv\n"
        f"{busiest_line}"
        f"{plot_line}"
        f"Trajectory plot:     {beam_plot}\n"
        f"Top {args.top_n} busiest {what}: {top_n_path}\n"
    )
    summary_path.write_text(summary)
    print(f"\n{summary}\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
