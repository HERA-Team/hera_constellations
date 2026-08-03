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
    - Text file (<outfile stem>_fov_times.txt): for each satellite, the UTC
      timestamp it first enters the beam and the timestamp of its last
      in-beam sample before leaving, i.e. the start/end of its FOV crossing
      in this window. Computed from beam membership regardless of
      --all-positions, so it's always the true entry/exit time, not just
      the first/last propagated sample.

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

--all-positions disables the beam-membership filter, plotting every
night-time position of every satellite instead (the old rafas.py
behaviour) - useful for comparing side by side against the default
beam-filtered plot.

To reuse for a different site/beam, edit parameters.py (HERA location, beam
radius) - shared with find_fov_passes.py, rather than a second copy here.
"""

import argparse
from functools import partial
from pathlib import Path

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

from tqdm import tqdm         # progress bar over satellites
from multiprocessing import Pool, RLock   # one worker process per satellite

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import colors
from matplotlib.cm     import ScalarMappable

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
        help="Disable the tqdm progress bar over satellites (useful for "
             "Slurm/batch jobs, where in-place redraws just clutter the "
             "log file).",
    )
    parser.add_argument(
        "--all-positions", action="store_true",
        help="Plot every propagated night-time position instead of only "
             "the ones inside the beam - reproduces the old rafas.py "
             "behaviour (full tracks, no beam-membership filtering), for "
             "comparing against the default beam-filtered plot. The beam "
             "outline is still drawn for reference; only the filtering of "
             "satellite points changes.",
    )
    return parser.parse_args()


#  Worker Function
def work_func_sat(sat_data, night_ts, in_beam_only=True):
    """Propagate one satellite and return the data of interest.

    Run in its own process per satellite - sat_data is the (name, tle1, 
    tle2) tuple for that satellite. 
    
    Thin wrapper around utils.propagate_in_beam(), which converts each 
    returned sample to RA/Dec (GCRS, not ICRS) and geodetic position.

    in_beam_only: passed straight through to propagate_in_beam() - True
    (default) yields only in-beam samples, False yields every night-time
    sample regardless of beam membership (--all-positions mode). 

    """
    name, tle1, tle2 = sat_data
    satellite_name = name.strip()
    times = night_ts['time']

    heights, latitudes, longitudes = [], [], []
    x, y, z = [], [], []
    ra, dec = [], []
    idx = []  # position of each hit in the original `times`, for date_nums[idx] later
    # First/last timestamp actually inside the beam - tracked from zen_sep
    # directly (not from `idx`/in_beam_only) so it's correct even in
    # --all-positions mode, where every night-time sample is yielded/kept,
    # not just in-beam ones. `times` is chronological, so the first/last
    # in-beam sample encountered here is the true entry/exit time.
    beam_entry, beam_exit = None, None

    for sample_idx, t1, itrs, altaz, zen_sep in propagate_in_beam(
        satellite_name, tle1, tle2, times, hera_location, fov_angular_radius,
        in_beam_only=in_beam_only,
    ):
        if zen_sep <= fov_angular_radius:
            if beam_entry is None:
                beam_entry = t1
            beam_exit = t1

        # In-beam: convert to RA/Dec for plotting on the sky. Use only the
        # alt/az *direction* (no distance) before transforming to GCRS -
        # GCRS is geocentric, so transforming `altaz` as-is (its real,
        # finite slant range) would re-centre it on Earth's core, introducing
        # a large parallax shift for near-Earth objects (slant range
        # comparable to Earth's radius) that the beam ring doesn't have (it's
        # built the same way but with no distance, i.e. treated as being at
        # infinity, like a star). Dropping the distance keeps both directly
        # comparable: the plotted point ends up at the same fixed-sky
        # direction that zen_sep already measures topocentrically, rather
        # than the (parallax-shifted) direction to the satellite's actual 3D
        # position as seen from Earth's centre.
        altaz_dir = SkyCoord(alt=altaz.alt, az=altaz.az, frame='altaz',
                              location=hera_location, obstime=t1)
        sky = altaz_dir.transform_to(GCRS(obstime=t1))
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
            np.array(ra), np.array(dec), np.array(idx, dtype=int),
            beam_entry, beam_exit)


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
    # full-TLE-file case (5 s) 
    # --time-delta still overrides either way.
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

    # TLE file
    sats = read_tle_file(args.tle_file)
    n_total = len(sats)

    if args.from_passes_dir:
        in_window_names = satellites_in_window(args.from_passes_dir, t0.iso, t_end.iso)
        sats = [s for s in sats if s[0].strip() in in_window_names]
        if not sats:
            raise SystemExit(
                f"--from-passes-dir {args.from_passes_dir}: no satellite has a "
                f"recorded pass between {t0.iso} and {t_end.iso}. Check the "
                "directory or decrease the time interval used to compute positions."
            )
        print(f"--from-passes-dir: {len(sats)}/{n_total} satellites have a "
              f"pass in this window; propagating only those.")

    # --limit caps how many are read, for a quick test run; default is every
    # satellite kept above.
    if args.limit is not None:
        sats = sats[:args.limit]
    print(f"Number of satellites: {len(sats)}")

    tqdm.set_lock(RLock())
    with Pool(4, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as pool:
        results = list(tqdm(
            pool.imap_unordered(
                partial(work_func_sat, night_ts=night_ts,
                        in_beam_only=not args.all_positions),
                sats,
            ),
            total=len(sats), desc="Satellites", disable=args.no_progress,
        ))

    # --- FOV entry/exit times -------------------------------------------
    # For each satellite, the UTC timestamp it first entered the beam and
    # its last in-beam sample before leaving, from beam_entry/beam_exit
    # computed in work_func_sat (regardless of --all-positions).
    fov_times_path = Path(args.outfile).with_name(Path(args.outfile).stem + "_fov_times.txt")
    with open(fov_times_path, "w", encoding="utf-8") as f:
        f.write(f"FOV entry/exit times, {t0.iso} to {t_end.iso}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Satellite':<20}{'First in FOV (UTC)':<27}{'Last in FOV (UTC)':<27}\n")
        for satname, _, _, _, _, _, beam_entry, beam_exit in sorted(
            results, key=lambda r: r[0]
        ):
            if beam_entry is None:
                f.write(f"{satname:<20}never in FOV during this window\n")
            else:
                f.write(f"{satname:<20}{beam_entry.iso:<27}{beam_exit.iso:<27}\n")
    print(f"Saved FOV entry/exit times to {fov_times_path}")

    # Build FoV outline (zenith radius 5 deg) for each time stamp.
    # Centre = HERA's zenith, transformed AltAz -> GCRS the same way satellite
    # positions are (see the import comment for why GCRS, not ICRS), so the
    # outline lines up with the satellite track frame (using LST directly).
    fov_ra, fov_dec = [], []
    centre_ra_wrapped = []
    for t in night_ts['time']:
        zenith = SkyCoord(
            alt=90*units.deg, az=0*units.deg, frame='altaz',
            location=hera_location, obstime=t
        )
        centre = zenith.transform_to(GCRS(obstime=t))
        ring = centre.directional_offset_by(position_angles, fov_angular_radius)
        centre_ra_wrapped.append(centre.ra.deg)
        fov_ra.append(ring.ra.deg)  # still wrapped into [0,360) here - fixed below
        fov_dec.append(ring.dec.deg)

    # Make RA continuous (allowed to run past 360 deg or below 0 deg)
    # instead of wrapping into [0,360): a night's beam drift is one
    # continuous sweep, but wrap_at(360) folds it back whenever it crosses
    # the 0/360 seam - which for ax.fill() below means two *adjacent*
    # vertices of the same ring can land on opposite sides of the seam
    # (e.g. 359.5 next to 0.3), and matplotlib just draws a straight edge
    # between them with no concept of the wrap, producing a spurious edge
    # stretching almost the full width of the plot. Fix: unwrap the (one
    # value per timestamp) centre RA across the whole night with
    # np.unwrap - safe since consecutive timestamps are only a tiny
    # fraction of a degree apart - then re-express every ring's vertices
    # as a small, always-well-behaved offset from that continuous centre,
    # rather than trying to patch up each ring's own wrapped vertex loop.
    centre_ra_wrapped = np.array(centre_ra_wrapped)
    centre_ra_unwrapped = np.degrees(np.unwrap(np.radians(centre_ra_wrapped)))
    for i in range(len(fov_ra)):
        offset = ((fov_ra[i] - centre_ra_wrapped[i] + 180) % 360) - 180
        fov_ra[i] = centre_ra_unwrapped[i] + offset

    # Satellite RA gets the same treatment, keyed by each point's `idx`
    # (its position in night_ts, matching centre_ra_wrapped/_unwrapped's own
    # indexing) - so satellite tracks and the FoV shading share one
    # continuous RA axis instead of the satellite dots staying wrapped
    # while the shading doesn't.
    results = [
        (satname, lat, lon,
         centre_ra_unwrapped[idx] + (((ra_arr - centre_ra_wrapped[idx] + 180) % 360) - 180),
         dec_arr, idx, beam_entry, beam_exit)
        for satname, lat, lon, ra_arr, dec_arr, idx, beam_entry, beam_exit in results
    ]

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

    # Normalize & colormap for shared mapping
    cmap = plt.get_cmap('coolwarm')
    norm = colors.Normalize(vmin=date_nums.min(), vmax=date_nums.max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # Draw FoV with time‐coded colors
    for ra_f, dec_f, tn in zip(fov_ra, fov_dec, date_nums):
        ax.fill(
            ra_f, dec_f,
            facecolor=cmap(norm(tn)), edgecolor='none',
            alpha=0.05, zorder=0
        )
    # # FoV polygons
    # for ra_fov, dec_fov in zip(fov_ra, fov_dec):
    #     ax.fill(ra_fov, dec_fov, edgecolor='none', alpha=0.009,
    #             color="teal", zorder=0)

    # satellite tracks. Colour by each point's own timestamp (date_nums[idx],
    # not date_nums[:len(ra_arr)] which paired points with the wrong times
    # once they were filtered down to in-beam samples), and share one
    # cmap/norm across satellites so the colour scale is consistent and the
    # colorbar below doesn't depend on which satellite happens to be plotted
    # last (some have zero in-beam points).
    for satname, lat, lon, ra_arr, dec_arr, idx, _, _ in results:
        ax.scatter(ra_arr, dec_arr,
                   c=date_nums[idx], marker='o', zorder=10,
                   cmap=cmap, norm=norm, s=25, edgecolors='black', linewidths=0.5,
                   label=satname)

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
    title_suffix = " (all positions, no beam filter)" if args.all_positions else ""
    # night_ts[0]/[-1] (not t0/t_end): --start/--end is often a deliberately
    # padded request window (e.g. run_pipeline.py passes a noon-to-noon span
    # for a --season busiest-night plot) - showing the actual night-filtered
    # bounds here means the title always reflects what was really plotted,
    # not the raw, wider request.
    ax.set_title(f"{args.constellation} Passes through HERA FoV{title_suffix}\n"
                 f"{night_ts['time'][0].iso} to {night_ts['time'][-1].iso}")
    ax.grid(alpha=0.5)

    # Report which satellites actually show up in the cropped view.
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    print(f"Satellites with a track visible in this plot:")
    for satname, lat, lon, ra_arr, dec_arr, idx, _, _ in results:
        if len(ra_arr) == 0:
            continue
        visible = ((ra_arr >= xlim[0]) & (ra_arr <= xlim[1]) &
                   (dec_arr >= ylim[0]) & (dec_arr <= ylim[1]))
        if visible.any():
            print(f"  {satname}, at RA={np.mean(ra_arr):.1f}deg, dec={np.mean(dec_arr):.1f}deg")

    # Shared colorbar built from the cmap/norm used above (rather than from
    # any one satellite's scatter handle), so it doesn't depend on plot order.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04,
                        label='UTC Time')
    cbar.ax.yaxis.set_major_locator(locator)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.savefig(args.outfile)
    print(f"Saved plot to {args.outfile}")
