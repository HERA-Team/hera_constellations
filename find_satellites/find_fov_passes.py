"""
find_fov_passes.py

Batch FOV-pass finder: reads a file of many satellites' TLEs, propagates
each one (in its own worker process) with SGP4 across a given observation
window, restricted to night-time hours at HERA (Sun below the local
horizon), and writes out every UTC timestamp at which the satellite is
inside HERA's field of view (FOV) - a circular disk of angular radius
`fov_angular_radius` centred on HERA's zenith (HERA is a drift-scan
telescope, so its beam center just sits at zenith and drifts across the
sky in RA as Earth rotates).

Outputs (written to --outdir, the current working directory by default):
    - <SATELLITE_NAME>_results.csv: a few '#'-prefixed comment lines
      (satellite name, TLE used) followed by a CSV header and one data row
      per night-time sample when the satellite was inside HERA's FOV:
      timestamp,alt_deg,az_deg,zensep_deg (zensep = angular distance from
      HERA's zenith). No rows means the satellite never passed through the
      FOV in this time range. One independent file per satellite (rather
      than one shared file/dataframe) so a worker crash or job preemption
      only loses that satellite's output, and so satellites already done
      don't need to be re-propagated on a re-run.
    - Running_details.txt: the run's physical parameters (time range, FOV,
      time step, HERA location - shared by every satellite, so recorded
      once here rather than repeated in every file), plus wall-clock time,
      peak memory (RSS) and other resource-usage stats for the whole run
      (useful for batch/cluster jobs where you need to check --time/--mem
      requests).

Usage:
    python find_fov_passes.py [tle_file] --day 2022-11-13 [--outdir OUTDIR]
    python find_fov_passes.py [tle_file] --season 2022 [--outdir OUTDIR]

    tle_file defaults to "tle_sample.txt" in the current directory if not
    given. It must list satellites in the standard 3-line format (name,
    TLE line 1, TLE line 2), blank lines are ignored - see
    utils.read_tle_file().
    --outdir defaults to the current directory, and is created if missing.

    Exactly one of --day/--season is required, to set the observation
    window (both are then restricted to night-time, Sun below horizon, by
    the code below):
    --day DATE     a single night: e.g. --day 2022-11-13 covers the night that
                   starts at dusk on Nov 13 and ends at dawn on Nov 14.
    --season YEAR  a full observing season: Sept 1 of YEAR to Apr 30 of
                   YEAR+1, e.g. --season 2022 for 2022-09-01 - 2023-04-30.
"""

import argparse
import os
import time
import psutil
import resource
from functools import partial

from astropy.time import Time
from astropy import units

from tqdm import tqdm         # per-satellite progress bar

from multiprocessing import Pool, RLock, current_process

from parameters import hera_lat, hera_lon, hera_alt, hera_location, fov_angular_radius
from utils import read_tle_file, make_time_grid, filter_to_night, propagate_in_beam

start = time.time()

# multiprocessing.Pool (spawn start method, the macOS/Windows default)
# re-imports this whole file in every worker process, re-running all
# module-level code below - including this print and the "Observation time
# considered" block further down. night_ts etc. genuinely need to be
# recomputed in each worker (that's why they're at module level rather than
# behind `if __name__ == "__main__"`), but printing about it doesn't - so
# gate the prints on being the original process, not a spawned worker, to
# avoid one copy of this output per worker.
is_main_process = current_process().name == "MainProcess"

if is_main_process:
    print("Start running code")


# --------------------------- Argument parsing ---------------------------
# Parsed up front (rather than only inside `if __name__ == "__main__"`)
# because --day/--season feed directly into the Global parameters section
# below, which needs to run at import time (night_ts etc. are module-level
# globals the worker function relies on).

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch FOV-pass finder: propagate many satellites' TLEs "
                     "with SGP4 and record every night-time timestamp and "
                     "position at which each one is inside HERA's FOV.",
    )
    parser.add_argument(
        "tle_file", nargs="?", default="tle_sample.txt",
        help="TLE file listing satellites in the standard 3-line format "
             "(name, TLE line 1, TLE line 2), blank lines ignored. "
             "Defaults to 'tle_sample.txt' in the current directory.",
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
        help="Restrict to the single night starting on this date: noon UTC "
             "on DATE to noon UTC the next day (the night-time filter below "
             "then picks out the actual dusk-to-dawn span within that).",
    )
    window.add_argument(
        "--season", type=int, metavar="YEAR",
        help="Restrict to the full observing season from Sept 1 of YEAR to "
             "Apr 30 of YEAR+1, e.g. --season 2022 for 2022-09-01 - "
             "2023-04-30.",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the per-satellite tqdm progress bars (useful for "
             "Slurm/batch jobs, where in-place redraws just clutter the "
             "log file).",
    )
    return parser.parse_args()


args = parse_args()


# ------------------------- Global parameters -------------------------
# hera_lat/hera_lon/hera_alt/hera_location and fov_angular_radius come from
# parameters.py - the single source of truth shared with beam_vs_starlink.py,
# rather than a second hand-copied block kept in sync manually.

# Observation time range, from --day or --season (mutually exclusive,
# exactly one required - see parse_args above).
if args.day:
    # Noon-to-noon UTC around the requested date, so the single actual
    # night (dusk on --day to dawn the next day) falls entirely inside it
    # regardless of HERA's small UTC offset - the night-time filter below
    # then trims this down to the real dusk-to-dawn span.
    day_start = Time(args.day, scale="utc")
    t0 = day_start + 12 * units.hour
    t_end = day_start + 36 * units.hour
else:
    t0 = Time(f"{args.season}-09-01T00:00:00", format="isot", scale="utc")
    t_end = Time(f"{args.season + 1}-04-30T23:59:00", format="isot", scale="utc")

# Sampling step - change accordingly. Needs to be finer than the shortest
# possible pass through the FOV (as brief as ~7 s for a fast LEO satellite)
# so a pass isn't stepped over between samples.
time_delta = 50 * units.s
if args.day:
    time_delta = 5 * units.s

# Regularly-spaced grid of timestamps across the full window (before any
# night-time filtering).
ts = make_time_grid(t0, t_end, time_delta)

# Filter to only night-time samples (Sun below the local horizon at HERA).
night_ts = filter_to_night(ts, hera_location)

# night_ts is already time-ordered (built from a mask over the sorted `ts`
# grid), so its first/last entries are the actual dusk/dawn bounds - a
# tighter, more meaningful range than the padded t0/t_end window above.
if is_main_process:
    print(f"\nObservation time considered (night-time only, Sun below horizon on site): "
          f"\nfrom {night_ts['time'][0].iso} \nto {night_ts['time'][-1].iso}")
    print("\nFOV of HERA:", fov_angular_radius*2.)
    print(f"\nTime delta: {time_delta} ")


# ------------------------------ Worker ---------------------------------

def work_func_sat(name, tle_line1, tle_line2, outdir, no_progress=False):
    """Propagate one satellite and write every in-FOV timestamp + position to a CSV file.

    Run in its own process (see Pool.starmap in __main__), one call per
    satellite. Thin wrapper around utils.propagate_in_beam(), which does
    the actual SGP4/TEME/ITRS/AltAz work shared with beam_vs_starlink.py;
    this function's own job is just writing a CSV row per in-beam sample,
    alongside the Alt/Az position already computed for the FOV test (so
    downstream analysis - e.g. Find Busy Times/find_busy.py - doesn't need
    to re-propagate the TLE to recover it).
    """
    satellite_name = name.strip()
    output_filename = os.path.join(outdir, f"{satellite_name.replace(' ', '_')}_results.csv")

    # Pin this worker's bar to its own terminal line (position) instead of
    # all pool workers fighting over line 0, which is what garbles the
    # output when several tqdm bars run in different processes at once.
    # _identity[0] is the 1-based slot of this process in the Pool.
    worker_slot = current_process()._identity[0]
    times = tqdm(night_ts['time'], desc=satellite_name,
                 position=worker_slot, leave=False, disable=no_progress)

    with open(output_filename, 'w') as f:
        # Metadata comment lines (skipped by csv readers/pandas via
        # comment='#'); run-wide parameters live in Running_details.txt
        # instead of being repeated in every satellite's file.
        f.write(f"# satellite: {satellite_name}\n")
        f.write(f"# tle_line1: {tle_line1}\n")
        f.write(f"# tle_line2: {tle_line2}\n")
        f.write("timestamp,alt_deg,az_deg,zensep_deg\n")

        n_visible = 0
        for _idx, t1, _itrs, altaz, zen_sep in propagate_in_beam(
            satellite_name, tle_line1, tle_line2, times, hera_location, fov_angular_radius
        ):
            n_visible += 1
            f.write(f"{t1.iso},{altaz.alt.to_value(units.deg):.6f},"
                    f"{altaz.az.to_value(units.deg):.6f},{zen_sep.to_value(units.deg):.6f}\n")

    # Print progress/result to screen. tqdm.write() (not plain print()) so
    # this doesn't land mid-line inside another worker's active progress
    # bar - plain print() isn't aware of tqdm's in-place cursor redraws.
    if n_visible:
        tqdm.write(f"\nFinished processing {satellite_name}, results in {output_filename}")
    else:
        tqdm.write(f"\nSatellite {name} does NOT pass through HERA's FOV in this time range.")


# ------------------------------- Main -----------------------------------

if __name__ == "__main__":
    satellites_data = read_tle_file(args.tle_file)

    os.makedirs(args.outdir, exist_ok=True)
    running_details = os.path.join(args.outdir, "Running_details.txt")

    # Share one lock across worker processes so their tqdm bars (each
    # pinned to its own line via `position` in work_func_sat) don't
    # interleave/corrupt each other's terminal writes.
    tqdm.set_lock(RLock())
    with Pool(initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        pool.starmap(
            partial(work_func_sat, outdir=args.outdir, no_progress=args.no_progress),
            satellites_data,
        )

    # --------------------- Resource usage report -------------------------
    # Wall time, memory and CPU stats for the whole run (self = this
    # process, children = the worker pool) - handy for checking batch/HPC
    # --time= and --mem= requests.
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
