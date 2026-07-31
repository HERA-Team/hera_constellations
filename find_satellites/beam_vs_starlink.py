"""
beam_vs_starlink.py

Propagates many satellites (e.g. a Starlink shell) from a TLE file using
SGP4, keeps only the positions that fall inside HERA's field of view (beam)
during night-time hours, and plots those in-beam points on the sky
(RA/Dec) together with the beam's drifting outline. Used to see how often
and where satellites cross HERA's beam over an observing run, which is
when they can contaminate 21cm observations.

The beam is modelled as a circular disk of fixed angular radius
(`fov_angular_radius`) centred on HERA's zenith - HERA is a drift-scan
telescope, so the beam center stays at zenith and drifts across the sky in
RA as Earth rotates, while its declination stays fixed at HERA's latitude.

Outputs:
    - Console: number of satellites read from the TLE file.
    - Figure: RA/Dec scatter of in-beam satellite positions colored by UTC
      time, overlaid on the drifting beam outline (--outfile,
      beam_vs_starlink.png by default).

Usage:
    python beam_vs_starlink.py TLE_FILE
    python beam_vs_starlink.py TLE_FILE --start 2022-11-13T21:45:00 \
        --end 2022-11-13T21:55:00 --outfile busiest_window.png
    python beam_vs_starlink.py TLE_FILE --start ... --end ... \
        --from-passes-dir path/to/find_fov_passes_outdir

With no --start/--end, defaults to a built-in ~10 min "busiest period"
preset (2022-11-13T21:45:10 to 21:55:10). Override both to point at a
different window, e.g. one found by Find Busy Times/find_busy.py.

--from-passes-dir speeds this up a lot for a short window over a big TLE
file (e.g. the full Starlink constellation): instead of propagating every
satellite just to find the few that are actually in the beam, it first
checks find_fov_passes.py's existing <SAT>_results.csv output for which
satellites have a pass in --start/--end, and only propagates those.

To reuse for a different site/beam, edit parameters.py (HERA location, beam
radius) - shared with find_fov_passes.py, rather than a second copy here.
"""

import argparse

import numpy as np

from astropy import units
from astropy.time import Time
# GCRS: the sky frame (RA/Dec) satellite/beam positions are converted to for
#   plotting - GCRS, not ICRS: ICRS applies a stellar-aberration correction
#   that assumes the object is effectively at infinite distance, which is
#   wrong for a satellite ~600-2000 km away (finite `distance` inherited
#   from AltAz) and silently gives a wildly wrong RA/Dec (tens of degrees
#   off) with no error. GCRS correctly handles the topocentric->geocentric
#   step for near-Earth distances and shares ICRS's orientation to well
#   under an arcsecond, so it's the right frame for both the satellite and
#   the beam outline.
from astropy.coordinates import AltAz, GCRS, SkyCoord

from tqdm import tqdm         # progress bar for the propagation loop
from multiprocessing import Pool, RLock, current_process   # one worker process per satellite

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors

from parameters import hera_location, fov_angular_radius, position_angles
from utils import (
    read_tle_file, make_time_grid, filter_to_night, propagate_in_beam,
    satellites_in_window,
)

# Built-in fallback window, used only if --start/--end aren't given: a
# known ~10 min "busiest" period with many overlapping passes, sampled
# finely enough (1 s) that a short pass (as brief as ~7 s across the beam)
# isn't stepped over between samples.
DEFAULT_START = "2022-11-13T21:45:10"
DEFAULT_END = "2022-11-13T21:55:10.000"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "tle_file",
        help="TLE file listing satellites in the standard 3-line format "
             "(name, TLE line 1, TLE line 2), blank lines ignored.",
    )
    parser.add_argument(
        "--start", default=None,
        help=f"Start of the time window (UTC, ISO format). Defaults to "
             f"{DEFAULT_START} if neither --start nor --end is given.",
    )
    parser.add_argument(
        "--end", default=None,
        help=f"End of the time window (UTC, ISO format). Defaults to "
             f"{DEFAULT_END} if neither --start nor --end is given.",
    )
    parser.add_argument(
        "--time-delta", type=float, default=None,
        help="Sampling cadence, in seconds. Defaults to 5 s, or 1 s if "
             "--from-passes-dir is given (matches the granularity needed "
             "so a ~7 s beam crossing isn't stepped over, for the handful "
             "of satellites a passes-dir run actually propagates).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only propagate the first N satellites from the TLE file "
             "(for a quick test run). Default: all satellites.",
    )
    parser.add_argument(
        "--from-passes-dir", default=None, metavar="DIR",
        help="Directory of find_fov_passes.py's <SAT>_results.csv output "
             "(its 'outdir'). If given, only satellites with at least one "
             "recorded pass inside --start/--end are propagated here, "
             "skipping the rest of the TLE file entirely - useful since "
             "most satellites never cross the beam in a given short "
             "window. Trajectories are still freshly propagated at "
             "--time-delta resolution for whichever satellites are kept "
             "(the CSVs are only used to pick which ones, not read for "
             "positions - they're coarser-cadence than a close-up window "
             "usually wants). Defaults --time-delta to 1 s instead of 5 s "
             "when set, since the point is a close look at few satellites.",
    )
    parser.add_argument(
        "--constellation", default="Satellite",
        help="Constellation name, used only for the default plot title "
             "(default: 'Satellite', giving 'Satellite Passes through "
             "HERA FoV').",
    )
    parser.add_argument(
        "--outfile", default="beam_vs_starlink.png",
        help="Output plot file path (default: beam_vs_starlink.png).",
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable the per-satellite tqdm progress bars (useful for "
             "Slurm/batch jobs, where in-place redraws just clutter the "
             "log file).",
    )
    return parser.parse_args()


#  Worker Function
def work_func_sat(name, tle1, tle2, night_ts, no_progress=False):
    """Propagate one satellite and return only the samples inside HERA's beam.

    Run in its own process (see Pool.starmap below), one call per satellite.
    Thin wrapper around utils.propagate_in_beam(), which does the actual
    SGP4/TEME/ITRS/AltAz work shared with find_fov_passes.py; this
    function's own job is converting each in-beam sample to RA/Dec (GCRS,
    not ICRS - see the import comment above for why) and geodetic
    position, and collecting the results.

    propagate_in_beam() only yields samples already inside the beam, so
    unlike the old NaN-prefill-and-mask approach, this just accumulates
    hits directly - no need to pre-size arrays for the full time range.

    Takes the whole night_ts TimeSeries (not just night_ts['time']) and
    extracts the column here, inside the worker, after it's been pickled
    across to this process - a sliced-out mixin column pickles badly (it
    carries a weakref back to its parent table), while the TimeSeries
    itself pickles fine.
    """
    satellite_name = name.strip()
    times = night_ts['time']

    heights, latitudes, longitudes = [], [], []
    x, y, z = [], [], []
    ra, dec = [], []
    idx = []  # position of each hit in the original `times`, for date_nums[idx] later

    # Pin this worker's bar to its own terminal line (position) instead of
    # all pool workers fighting over line 0, which is what garbles the
    # output when several tqdm bars run in different processes at once.
    # _identity[0] is the 1-based slot of this process within the Pool.
    worker_slot = current_process()._identity[0]
    progress_times = tqdm(times, desc=satellite_name,
                           position=worker_slot, leave=False, disable=no_progress)

    for sample_idx, t1, itrs, altaz, _zen_sep in propagate_in_beam(
        satellite_name, tle1, tle2, progress_times, hera_location, fov_angular_radius
    ):
        # In-beam: convert to RA/Dec for plotting on the sky.
        sky = altaz.transform_to(GCRS(obstime=t1))
        loc = itrs.earth_location

        heights.append(loc.height.to(units.km).value)
        latitudes.append(loc.lat.deg)
        longitudes.append(loc.lon.deg)
        x.append(loc.x.to(units.km).value)
        y.append(loc.y.to(units.km).value)
        z.append(loc.z.to(units.km).value)
        ra.append(sky.ra.deg)
        dec.append(sky.dec.deg)
        idx.append(sample_idx)

    # idx (positions in the original `times`) is returned alongside the
    # other arrays so the caller can pick out the matching timestamps
    # (e.g. date_nums[idx]) for plotting.
    return (satellite_name, np.array(latitudes), np.array(longitudes),
            np.array(ra), np.array(dec), np.array(idx, dtype=int))


#
# Main
if __name__ == "__main__":
    args = parse_args()

    if args.start or args.end:
        start_str = args.start or DEFAULT_START
        end_str = args.end or DEFAULT_END
    else:
        start_str, end_str = DEFAULT_START, DEFAULT_END
    t0 = Time(start_str, format="isot", scale="utc")
    t_end = Time(end_str, format="isot", scale="utc")

    # --from-passes-dir narrows things down to a handful of satellites for a
    # close-up window, so default to finer (1 s) cadence than the plain
    # full-TLE-file case (5 s) - --time-delta still overrides either way.
    default_time_delta = 1.0 if args.from_passes_dir else 5.0
    time_delta = (args.time_delta if args.time_delta is not None
                  else default_time_delta) * units.s

    # Regularly-spaced grid of timestamps across the full window (before any
    # night-time filtering).
    ts = make_time_grid(t0, t_end, time_delta)

    # Filter to only night-time samples (Sun below the local horizon at HERA).
    night_ts = filter_to_night(ts, hera_location)

    if len(night_ts) == 0:
        raise SystemExit(
            f"No night-time samples between {t0.iso} and {t_end.iso} at HERA - "
            "the Sun is up the whole window. Pick a different --start/--end."
        )

    # TLE file - each satellite is propagated in its own process (Pool),
    # since the per-satellite work is independent.
    sats = read_tle_file(args.tle_file)
    n_total = len(sats)

    if args.from_passes_dir:
        in_window_names = satellites_in_window(args.from_passes_dir, t0.iso, t_end.iso)
        sats = [s for s in sats if s[0].strip() in in_window_names]
        if not sats:
            raise SystemExit(
                f"--from-passes-dir {args.from_passes_dir}: no satellite has a "
                f"recorded pass between {t0.iso} and {t_end.iso}. Check the "
                "directory is find_fov_passes.py's outdir for this same TLE "
                "file and time range."
            )
        print(f"--from-passes-dir: {len(sats)}/{n_total} satellites have a "
              f"pass in this window; propagating only those.")

    # --limit caps how many are read, for a quick test run; default is every
    # satellite kept above.
    if args.limit is not None:
        sats = sats[:args.limit]
    print(f"Number of satellites: {len(sats)}")

    # Share one lock across worker processes so their tqdm bars (each
    # pinned to its own line via `position` in work_func_sat) don't
    # interleave/corrupt each other's terminal writes.
    tqdm.set_lock(RLock())
    with Pool(4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = pool.starmap(
            work_func_sat,
            [(name, tle1, tle2, night_ts, args.no_progress)
             for name, tle1, tle2 in sats],
        )

    # Build FoV outline (zenith radius 5 deg) for each time stamp.
    # Centre = HERA's zenith, transformed AltAz -> GCRS the same way satellite
    # positions are (see the import comment for why GCRS, not ICRS), so the
    # outline lines up with the satellite track frame (using LST directly as
    # an RA ignores ~20+ years of precession since J2000, which is a
    # sizeable fraction of a 10 deg FoV).
    fov_ra, fov_dec = [], []
    for t in night_ts['time']:
        # SkyCoord (not a bare AltAz frame) so .transform_to() below yields a
        # SkyCoord too - directional_offset_by is a SkyCoord-only method,
        # not available on plain frame objects.
        zenith = SkyCoord(alt=90*units.deg, az=0*units.deg, frame='altaz',
                           location=hera_location, obstime=t)
        centre = zenith.transform_to(GCRS(obstime=t))
        ring   = centre.directional_offset_by(position_angles, fov_angular_radius)
        fov_ra.append(ring.ra.wrap_at(360*units.deg).deg)
        fov_dec.append(ring.dec.deg)

    # Night filtering can produce disjoint time blocks (e.g. an early-morning
    # block and a separate evening block), so the FoV drift isn't one
    # contiguous arc - take min/max over all points, not just the first/last
    # timestamp's ring.
    all_fov_ra  = np.concatenate(fov_ra)
    all_fov_dec = np.concatenate(fov_dec)
    ra_min, ra_max   = all_fov_ra.min(),  all_fov_ra.max()
    dec_min, dec_max = all_fov_dec.min(), all_fov_dec.max()

    # PLOT
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    # Convert times to matplotlib's numeric date format once, so it can be
    # reused as the colorbar data and indexed per-satellite (date_nums[idx]).
    times      = night_ts['time'].to_datetime()
    date_nums  = mdates.date2num(times)
    # common locator/formatter for the "UTC Time" colorbar (nicely-spaced date ticks)
    locator    = mdates.AutoDateLocator()
    formatter  = mdates.ConciseDateFormatter(locator)

    # FoV polygons
    for ra_fov, dec_fov in zip(fov_ra, fov_dec):
        ax.fill(ra_fov, dec_fov, edgecolor='none', alpha=0.009,
                color="teal", zorder=0)

    # satellite tracks. Colour by each point's own timestamp (date_nums[idx],
    # not date_nums[:len(ra_arr)] which paired points with the wrong times
    # once they were filtered down to in-beam samples), and share one
    # cmap/norm across satellites so the colour scale is consistent and the
    # colorbar below doesn't depend on which satellite happens to be plotted
    # last (some have zero in-beam points).
    cmap = plt.get_cmap('coolwarm')
    norm = colors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
    for satname, lat, lon, ra_arr, dec_arr, idx in results:
        ax.scatter(ra_arr, dec_arr,
                   c=date_nums[idx], marker='o',
                   cmap=cmap, norm=norm, s=1, lw=1.3, label=satname)

    # Not 'equal': the RA range covered here (up to ~360 deg, since the
    # night-time filter can produce blocks many hours apart, over which
    # HERA's zenith RA sweeps most of the sky) vastly exceeds the Dec
    # range (~2*fov_angular_radius, fixed by the beam). An equal aspect
    # would force 1 deg RA == 1 deg Dec on screen and squash the whole
    # plot into an unreadable sliver.
    ax.set_aspect('auto')
    ax.set_xlabel("Right Ascension (deg)")
    ax.set_ylabel("Declination (deg)")
    # 2 deg padding around the full beam-drift extent so the outline isn't
    # cut off at the plot edges.
    ax.set_xlim(ra_min - 2, ra_max + 2)
    ax.set_ylim(dec_min - 2, dec_max + 2)
    ax.set_title(f"{args.constellation} Passes through HERA FoV\n"
                 f"{t0.iso} to {t_end.iso}")
    ax.grid(alpha=0.5)

    # Shared colorbar built from the cmap/norm used above (rather than from
    # any one satellite's scatter handle), so it doesn't depend on plot order.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04,
                        label='UTC Time')
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.savefig(args.outfile)
    print(f"Saved plot to {args.outfile}")
