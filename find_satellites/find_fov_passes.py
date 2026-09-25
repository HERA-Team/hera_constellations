"""
find_fov_passes.py

Batch FOV-pass finder: reads a file of many satellites' TLEs, propagates
each one with SGP4 across a given observation
window, restricted to night-time hours at HERA (Sun below the local
horizon), and writes out every UTC timestamp at which the satellite is
inside HERA's field of view, taken to be a circular disk of angular radius
`fov_angular_radius` centred on HERA's zenith.

Outputs (written to --outdir, the current working directory by default):
    - <SATELLITE_NAME>_results.csv: a few '#'-prefixed comment lines
      (satellite name, TLE used, time_delta) followed by a CSV header and
      one data row per night-time sample when the satellite was inside
      HERA's FOV: timestamp,alt_deg,az_deg,zensep_deg (zensep = angular
      distance from HERA's zenith). No rows means the satellite never
      passed through the FOV in this time range. One independent file per
      satellite.
      If a satellite's CSV already exists and was generated with the same
      time_delta this run would use, it's skipped rather than
      re-propagated. Pass --overwrite to force recomputation regardless.

    - Running_details.txt: the run's physical parameters (time range, FOV,
      time step, HERA location), plus wall-clock time,
      peak memory (RSS) and other resource-usage stats for the whole run.

Usage:
    python find_fov_passes.py [tle_file] --day 2022-11-13 [--outdir OUTDIR]
    python find_fov_passes.py [tle_file] --season 2022 [--outdir OUTDIR]
    python find_fov_passes.py --tle-dir DAILY_TLE_DIR --season 2022 [--outdir OUTDIR]

    tle_file defaults to "tle_sample.txt" in the current directory if not
    given. It must list satellites in the standard 3-line format (name,
    TLE line 1, TLE line 2), blank lines are ignored - see
    utils.read_tle_file().

    --outdir defaults to the current directory, and is created if missing.

    Two options:
    --day DATE     a single night: e.g. --day 2022-11-13 covers the night that
                   starts at dusk on Nov 13 and ends at dawn on Nov 14.
    --season YEAR  a full observing season: Sept 1 of YEAR to Apr 30 of
                   YEAR+1, e.g. --season 2022 for 2022-09-01 - 2023-04-30.

    --tle-dir DIR replaces the single tle_file with a directory of daily
    TLE files in the --season mode. Each
    satellite is propagated with its own closest-available TLE per night.

    --time-delta overrides the sampling cadence in seconds (default: 5 s
    for --day, 50 s for --season).
"""

import argparse
import glob
import os
import re
import time
import psutil
import resource
from functools import partial
from pathlib import Path

from astropy.time import Time
from astropy import units

from tqdm import tqdm

from multiprocessing import Pool, RLock, current_process

from parameters import hera_lat, hera_lon, hera_alt, hera_location, fov_angular_radius
from utils import (
    read_tle_file, make_time_grid, filter_to_night, propagate_in_beam,
    propagate_in_beam_daily_tle,
)

start = time.time()
is_main_process = current_process().name == "MainProcess"

if is_main_process:
    print("Start running code")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch FOV-pass finder: propagate many satellites' TLEs "
                     "with SGP4 and record every night-time timestamp and "
                     "position at which each one is inside HERA's FOV.",
    )
    parser.add_argument(
        "tle_file", nargs="?", default=None,
        help="TLE file listing satellites in the standard 3-line format "
             "(name, TLE line 1, TLE line 2), blank lines ignored. "
             "Defaults to 'tle_sample.txt' in the current directory. "
             "Mutually exclusive with --tle-dir.",
    )
    parser.add_argument(
        "--tle-dir", default=None, metavar="DIR",
        help="Directory of daily TLE files (one <date>_<constellation>_"
             "tles.txt per day) in --season mode only.",
    )
    parser.add_argument(
        "--outdir", default=".",
        help="Directory to write <SATELLITE_NAME>_results.csv and "
             "Running_details.txt to (created if it doesn't exist). "
             "Defaults to the current directory.",
    )
    window = parser.add_mutually_exclusive_group(required=True)
    window.add_argument(
        "--day", metavar="YYYY-MM-DD",
        help="Restrict to the single night starting on this date.",
    )
    window.add_argument(
        "--season", type=int, metavar="YEAR",
        help="Restrict to the full observing season from Sept 1 of YEAR to "
             "Apr 30 of YEAR+1, e.g. --season 2022 for 2022-09-01 - "
             "2023-04-30.",
    )
    parser.add_argument(
        "--time-delta", type=float, default=None,
        help="Sampling cadence, in seconds. Defaults to 5 s for --day, or "
             "50 s for --season.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the tqdm progress bar over satellites.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute every satellite even if its <SAT>_results.csv "
             "already exists with a matching time_delta. By default, a "
             "satellite is skipped (not re-propagated) if its results file "
             "is already there.",
    )
    args = parser.parse_args()

    if args.tle_dir and args.day:
        parser.error("--tle-dir requires --season, not --day.")
    if args.tle_dir and args.tle_file is not None:
        parser.error("cannot combine a positional tle_file with --tle-dir.")
    if not args.tle_dir and args.tle_file is None:
        args.tle_file = "tle_sample.txt"  # restore the old default

    return args


args = parse_args()


# ------------------------- Global parameters -------------------------

# Observation time range
if args.day:
    # Noon-to-noon UTC around the requested date, so the single actual
    # night falls entirely inside it.
    day_start = Time(args.day, scale="utc")
    t0 = day_start + 12 * units.hour
    t_end = day_start + 36 * units.hour
else:
    t0 = Time(f"{args.season}-09-01T00:00:00", format="isot", scale="utc")
    t_end = Time(f"{args.season + 1}-04-30T23:59:00", format="isot", scale="utc")

# Sampling step - --time-delta overrides; otherwise 5 s for --day or 50 s
# for --season (needs to be finer than the shortest possible pass through
# the FOV, as brief as ~7 s for a fast LEO satellite, so a pass isn't
# stepped over between samples).
default_time_delta = 5.0 if args.day else 50.0
time_delta = (args.time_delta if args.time_delta is not None
              else default_time_delta) * units.s

# Regularly-spaced grid of timestamps across the full window (before any
# night-time filtering).
ts = make_time_grid(t0, t_end, time_delta)

# Filter to only night-time samples (Sun below the local horizon at HERA).
night_ts = filter_to_night(ts, hera_location)

if is_main_process:
    print(f"\nObservation time considered (night-time only, Sun below horizon on site): "
          f"\nfrom {night_ts['time'][0].iso} \nto {night_ts['time'][-1].iso}")
    print("\nFOV of HERA:", fov_angular_radius*2.)
    print(f"\nTime delta: {time_delta} ")

# --tle-dir: one TLE file per day instead of a single tle_file.
if args.tle_dir:
    tle_by_sat = {}
    for path in sorted(glob.glob(os.path.join(args.tle_dir, "*_tles.txt"))):
        match = re.match(r"^(\d{4}-\d{2}-\d{2})_", os.path.basename(path))
        if not match:
            print(f"Warning: {path} doesn't match <date>_<constellation>_tles.txt; skipping.")
            continue
        date_str = match.group(1)
        for name, tle_line1, tle_line2 in read_tle_file(path):
            tle_by_sat.setdefault(name.strip(), {})[date_str] = (tle_line1, tle_line2)
    if is_main_process:
        print(f"\n--tle-dir {args.tle_dir}: {len(tle_by_sat)} satellites across "
              f"{len(glob.glob(os.path.join(args.tle_dir, '*_tles.txt')))} daily TLE files.")
else:
    tle_by_sat = None


# ------------------------------ Worker ---------------------------------

def work_func_sat(sat_data, outdir):
    """Propagate one satellite and write every in-FOV timestamp + position to a CSV file.

    Run in its own process, one call per satellite.

    """
    if args.tle_dir:
        name = sat_data
        tle_by_date = tle_by_sat.get(name, {})
    else:
        name, tle_line1, tle_line2 = sat_data
    satellite_name = name.strip()
    output_filename = os.path.join(outdir, f"{satellite_name.replace(' ', '_')}_results.csv")
    time_delta_s = time_delta.to_value(units.s)
    # A single TLE pair doesn't mean anything in --tle-dir mode.
    expected_tle_source = str(Path(args.tle_dir).resolve()) if args.tle_dir else None

    if not args.overwrite and os.path.exists(output_filename):
        existing_delta_s = None
        existing_tle_source = None
        with open(output_filename) as f:
            for line in f:
                if not line.startswith("#"):
                    break  # comment header always comes first; nothing more to find
                if line.startswith("# time_delta_s:"):
                    try:
                        existing_delta_s = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif line.startswith("# tle_source:"):
                    existing_tle_source = line.split(":", 1)[1].strip()
        if existing_delta_s == time_delta_s and existing_tle_source == expected_tle_source:
            tqdm.write(f"\nSkipping {satellite_name}: {output_filename} already exists "
                       f"with matching time_delta={existing_delta_s} s and TLE source "
                       "(use --overwrite to force).")
            return

    with open(output_filename, 'w') as f:
        # Metadata comment lines
        f.write(f"# satellite: {satellite_name}\n")
        if args.tle_dir:
            f.write(f"# tle_source: {expected_tle_source}\n")
        else:
            f.write(f"# tle_line1: {tle_line1}\n")
            f.write(f"# tle_line2: {tle_line2}\n")
        f.write(f"# time_delta_s: {time_delta_s}\n")
        f.write("timestamp,alt_deg,az_deg,zensep_deg\n")

    if args.tle_dir:
        crossings = propagate_in_beam_daily_tle(
            satellite_name, tle_by_date, night_ts['time'],
            hera_location, fov_angular_radius,
        )
    else:
        crossings = propagate_in_beam(
            satellite_name, tle_line1, tle_line2, night_ts['time'],
            hera_location, fov_angular_radius,
        )

    n_visible = 0
    for _, t1, _, altaz, zen_sep in crossings:
        n_visible += 1
        with open(output_filename, 'a') as f:
            f.write(f"{t1.iso},{altaz.alt.to_value(units.deg):.6f},"
                    f"{altaz.az.to_value(units.deg):.6f},{zen_sep.to_value(units.deg):.6f}\n")


# ------------------------------- Main -----------------------------------

if __name__ == "__main__":
    satellites_data = sorted(tle_by_sat) if args.tle_dir else read_tle_file(args.tle_file)

    os.makedirs(args.outdir, exist_ok=True)
    running_details = os.path.join(args.outdir, "Running_details.txt")

    print(f'\nComputing satellite positions over {len(night_ts["time"])} timestamps '
          f'for {len(satellites_data)} satellites.')

    # Share one lock across worker processes so their tqdm.write() calls
    # don't corrupt the main process's satellite-count progress bar.
    tqdm.set_lock(RLock())
    n_workers = min(os.cpu_count() or 1, 16)
    with Pool(n_workers, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(work_func_sat, outdir=args.outdir), satellites_data
            ),
            total=len(satellites_data), desc="Satellites",
            disable=args.no_progress,
        ):
            pass

    # --------------------- Resource usage report -------------------------
    # Wall time, memory and CPU stats for the whole run (self = this
    # process, children = the worker pool)
    proc = psutil.Process(os.getpid())
    usage_self = resource.getrusage(resource.RUSAGE_SELF)
    usage_ch = resource.getrusage(resource.RUSAGE_CHILDREN)

    user_total = usage_self.ru_utime + usage_ch.ru_utime
    system_total = usage_self.ru_stime + usage_ch.ru_stime
    minflt_total = usage_self.ru_minflt + usage_ch.ru_minflt    # minor (reclaim) page faults
    majflt_total = usage_self.ru_majflt + usage_ch.ru_majflt    # major (I/O) page faults
    invol_total = usage_self.ru_nivcsw + usage_ch.ru_nivcsw     # involuntary context switches
    vol_total = usage_self.ru_nvcsw + usage_ch.ru_nvcsw         # voluntary context switches
    inb_total = usage_self.ru_inblock + usage_ch.ru_inblock     # block input operations
    outb_total = usage_self.ru_oublock + usage_ch.ru_oublock    # block output operations
    swaps_total = usage_self.ru_nswap + usage_ch.ru_nswap       # number of swaps

    # ru_maxrss is the *peak* RSS already, so take the max (not the sum)
    # across self and children.
    peak_rss_kb = max(usage_self.ru_maxrss, usage_ch.ru_maxrss)
    mem = proc.memory_info().rss / 1024**2   # current RSS of this process, in MB

    end = time.time()
    wall = end - start
    cpu_pct = 100.0 * (user_total + system_total) / wall

    print("\n")
    print(f"Elapsed wall time (Running time): {wall:.1f} s")
    print(f"Peak memory (RSS), from getrusage: {peak_rss_kb/1024:.1f} MB, psutil RSS: {mem:.1f} MB")
    print(f"CPU used: {user_total + system_total:.5f} s  ({cpu_pct:.3f}% of wall), cores =~ {cpu_pct/100:.2f}")

    with open(running_details, 'w') as f:
        f.write("Run Parameters \n")
        f.write("=============================================\n\n")
        f.write(f"Time considered (night-time only, Sun below horizon at HERA): "
                f"from {night_ts['time'][0].iso} to {night_ts['time'][-1].iso}\n")
        f.write(f"Effective FOV of HERA: {fov_angular_radius}\n")
        f.write(f"Time delta = {time_delta}, Number of night-time samples = {len(night_ts)}\n")
        f.write(f"HERA location: lat={hera_lat}, lon={hera_lon}, alt={hera_alt}\n\n")

        f.write("Code Running Details \n")
        f.write("=============================================\n\n")
        f.write(f"Elapsed wall time (Running time): {wall:.1f} s\n")
        f.write(f"Peak memory (RSS), from getrusage: {peak_rss_kb/1024:.1f} MB, psutil RSS: {mem:.1f} MB\n")
        f.write(f"CPU used: {user_total + system_total:.5f} s  ({cpu_pct:.3f}% of wall), cores =~ {cpu_pct/100:.2f}\n\n")

        f.write("Some other details\n")
        f.write("-----------------------------------\n")
        f.write(f"Maximum resident set size (kbytes): {max(usage_self.ru_maxrss, usage_ch.ru_maxrss)}\n")
        f.write(f"Major (requiring I/O) page faults:   {majflt_total}\n")
        f.write(f"Minor (reclaiming a frame) page faults: {minflt_total}\n")
        f.write(f"Voluntary context switches:   {vol_total}\n")
        f.write(f"Involuntary context switches: {invol_total}\n")
        f.write(f"File system inputs:  {inb_total}\n")
        f.write(f"File system outputs: {outb_total}\n")
        f.write(f"Swaps: {swaps_total}\n")

        print(f"Code running resource details printed to: {running_details} \n")
