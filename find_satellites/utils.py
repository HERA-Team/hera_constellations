"""
utils.py

Shared functions for the find_satellites/ scripts - previously duplicated
(or near-duplicated) between find_fov_passes.py and beam_vs_starlink.py.

Nothing here imports from parameters.py: every function takes `location`/
`fov_radius` as an explicit argument rather than reaching for a default, so
the dependency graph stays one-directional and these functions work for any
site/beam, not just the hardcoded HERA one in parameters.py.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from astropy import units
from astropy.time import Time
from astropy.timeseries import TimeSeries
# TEME: satellite-native True Equator Mean Equinox frame SGP4 propagates in.
# ITRS: Earth-fixed frame used as a stepping stone to the topocentric view.
# AltAz + get_sun: used both for the night-time filter and the FOV test.
from astropy.coordinates import (
    TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz, ICRS, get_sun,
)

from sgp4.api import Satrec   # SGP4 orbit propagator, reads TLEs


def read_tle_file(filename):
    """Parse a TLE file into a list of (name, tle_line1, tle_line2) tuples.

    Expects the standard 3-line-per-satellite format (name, then the two
    TLE lines), with blank lines ignored.
    """
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip() != ""]
    satellites = []
    for i in range(0, len(lines), 3):
        satellites.append((lines[i], lines[i + 1], lines[i + 2]))
    return satellites


def utc_to_jd(utc_time):
    """Convert a UTC date/time string to a Julian Date.

    Accepts anything astropy.time.Time parses under scale='utc', e.g.
    '2022-10-23T21:45:00.000' (isot) or '2022-10-23 21:45:00' (iso).
    """
    return Time(utc_time, scale="utc").jd


def jd_to_utc(jd):
    """Convert a Julian Date back to a UTC date/time string (isot format).

    Inverse of utc_to_jd(), e.g. 2459876.4059606 -> '2022-10-23T21:45:00.000'.
    """
    return Time(jd, format="jd", scale="utc").isot


def make_time_grid(t0, t_end, time_delta):
    """Build a regularly-spaced TimeSeries from t0 to t_end at time_delta cadence."""
    total_seconds = (t_end - t0).sec
    n_samples = int(total_seconds // time_delta.value) + 1
    return TimeSeries(time_start=t0, time_delta=time_delta, n_samples=n_samples)


def filter_to_night(ts, location):
    """Restrict a TimeSeries to samples where the Sun is below `location`'s horizon.

    sun_alt is the Sun's altitude angle at `location` for every timestamp in
    ts; negative altitude means the Sun is below the horizon (i.e. it's
    night).
    """
    sun_alt = get_sun(ts['time']).transform_to(
        AltAz(obstime=ts['time'], location=location)
    ).alt
    return TimeSeries(time=ts['time'][sun_alt < 0 * units.deg])


def propagate_in_beam(name, tle_line1, tle_line2, times, location, fov_radius,
                       in_beam_only=True):
    """Propagate one satellite across `times` and yield samples inside the beam.

    Builds its own SGP4 propagator from the TLE, then for each timestamp in
    `times` (iterable of astropy Time objects) runs the conversion: 
    SGP4 -> TEME -> ITRS -> AltAz, testing whether the result falls
    within `fov_radius` of `location`'s zenith. 
    Note: SGP4 errors are silently skipped.

    in_beam_only: if True (default), samples outside `fov_radius` of zenith
    are skipped too, so only true beam crossings are yielded. Pass
    False to yield every sample regardless of beam membership.

    Yields (idx, t1, itrs, altaz, zen_sep) 
    - idx is the position of t1 in the original `times` sequence 
    - itrs/altaz are the corresponding coordinate frame objects 
    - zen_sep is the angular distance from zenith
    """
    sat = Satrec.twoline2rv(tle_line1, tle_line2)

    for idx, t1 in enumerate(times):
        # Propagate the TLE to time t1: returns position/velocity in the
        # (quasi-inertial) TEME frame.
        error_code, teme_p, teme_v = sat.sgp4(t1.jd1, t1.jd2)
        if error_code != 0:
            continue  # SGP4 failed for this timestamp; skip and move on

        # Wrap the raw SGP4 outputs as astropy quantities, then convert
        # TEME -> ITRS, the Earth-fixed frame, as a stepping stone to the
        # topocentric (observer-relative) view below.
        pos = CartesianRepresentation(teme_p * units.km)
        vel = CartesianDifferential(teme_v * units.km / units.s)
        teme = TEME(pos.with_differentials(vel), obstime=t1)
        itrs = teme.transform_to(ITRS(obstime=t1))

        # ITRS -> AltAz as seen from `location` (not Earth's center).
        altaz = itrs.transform_to(AltAz(obstime=t1, location=location))

        # Angular distance from zenith
        # inside the FOV disk if within fov_radius of it.
        zen_sep = 90 * units.deg - altaz.alt
        if in_beam_only and zen_sep > fov_radius:
            continue  # outside the beam - skip

        yield idx, t1, itrs, altaz, zen_sep


def satellites_in_window(passes_dir, start_iso, end_iso):
    """Return the set of satellite names with >= 1 row inside [start_iso,
    end_iso] across find_fov_passes.py's <SAT>_results.csv files in
    passes_dir.

    start_iso/end_iso and the CSVs' timestamps are both plain
    'YYYY-MM-DD HH:MM:SS.sss' strings (astropy Time.iso format), which sort
    correctly with a plain string comparison - no need to parse them.
    Satellite name comes from each file's '# satellite: ...' comment line
    (the original name, spaces included) rather than the filename, since
    find_fov_passes.py replaces spaces with underscores in the filename.
    """
    names = set()
    for csv_path in Path(passes_dir).glob("*_results.csv"):
        sat_name = None
        with open(csv_path) as fh:
            for line in fh:
                if line.startswith("# satellite:"):
                    sat_name = line.split(":", 1)[1].strip()
                    continue
                if line.startswith("#") or line.startswith("timestamp,"):
                    continue
                ts = line.split(",", 1)[0]
                if start_iso <= ts <= end_iso:
                    if sat_name:
                        names.add(sat_name)
                    break  # one hit is enough for this file
    return names


def in_beam_mask(uv, location, fov_radius):
    """Boolean array, one entry per `uv`'s unique time, for whether its
    phase center was inside the FOV (within `fov_radius` of `location`'s
    zenith) at that time.

    Works directly off `uv.phase_center_catalog`, not a TLE or a
    find_fov_passes.py output directory, by transforming the phase
    center's own stored apparent position back to AltAz at each time and
    comparing the zenith separation to `fov_radius`. For an "ephem" phase
    center built by phasing to a satellite with propagate_in_beam() (e.g.
    via phase_data/phase_to_satellites.py), this recovers the same zenith
    separation this module's own propagate_in_beam() would have computed,
    since that ephemeris keeps only the satellite's topocentric
    *direction* (no distance) - exactly what's being inverted here.

    Handles a `uv` whose blts are split across more than one phase center
    (e.g. some samples phased to a satellite, others left at the original
    "unprojected"/zenith center) by looking up each unique time's own
    phase center id via `uv.phase_center_id_array`.
    """
    unique_jd, unique_idx = np.unique(uv.time_array, return_index=True)
    cat_ids = uv.phase_center_id_array[unique_idx]

    zensep_deg = np.zeros(unique_jd.shape)
    for cat_id in np.unique(cat_ids):
        cat = uv.phase_center_catalog[cat_id]
        sel = cat_ids == cat_id
        times = Time(unique_jd[sel], format="jd", scale="utc")

        if cat["cat_type"] == "ephem":
            # cat_times/cat_lon/cat_lat cover exactly the times this phase
            # center was applied to, so this interpolation lands on exact
            # matches, not an approximation.
            lon = np.interp(unique_jd[sel], cat["cat_times"], cat["cat_lon"])
            lat = np.interp(unique_jd[sel], cat["cat_times"], cat["cat_lat"])
        else:
            lon = np.full(sel.sum(), cat["cat_lon"])
            lat = np.full(sel.sum(), cat["cat_lat"])

        icrs = ICRS(ra=lon * units.rad, dec=lat * units.rad)
        altaz = icrs.transform_to(AltAz(obstime=times, location=location))
        zensep_deg[sel] = (90 * units.deg - altaz.alt).to_value(units.deg)

    return zensep_deg <= fov_radius.to_value(units.deg)


def plot_waterfall(uv, location, fov_radius, ant1=None, ant2=None, pol=None,
                    ax=None, shade_in_beam=True):
    """Plot a time-vs-frequency amplitude waterfall for one baseline/pol of `uv`.

    If shade_in_beam (default True), the times `uv`'s phase center was
    actually inside the FOV are hatched - see in_beam_mask() above. For a
    `uv` phased with in_beam_only=True (find_fov_passes.py's default
    sense, and phase_data/phase_to_satellites.py's), every sample is
    in-beam, so the whole plot is hatched; the shading is most
    informative for a `uv` phased with in_beam_only=False, which also
    includes samples before/after the FOV crossing.

    ant1/ant2: baseline to plot; defaults to `uv`'s first antenna pair.
    pol: polarization to plot (str or int, anything uv.get_data() accepts);
        defaults to `uv`'s first polarization.
    ax: existing matplotlib Axes to draw on; a new figure is created if
        not given.

    Returns the Axes drawn on.
    """
    if ant1 is None or ant2 is None:
        ant1, ant2 = uv.get_antpairs()[0]
    if pol is None:
        pol = uv.get_pols()[0]

    data = uv.get_data(ant1, ant2, pol)  # (Ntimes, Nfreqs)
    times = np.unique(uv.time_array)  # matches get_data()'s time ordering
    freqs_mhz = uv.freq_array / 1e6
    t_minutes = (times - times[0]) * 24 * 60  # minutes since first sample

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    im = ax.pcolormesh(freqs_mhz, t_minutes, np.abs(data), shading="nearest")
    plt.colorbar(im, ax=ax, label="|Visibility|")

    if shade_in_beam:
        in_beam = in_beam_mask(uv, location, fov_radius)
        # Hatch each contiguous in-beam run as a horizontal band spanning
        # the full frequency range, rather than solid-filling it (which
        # would hide the data underneath).
        padded = np.concatenate(([0], in_beam.astype(int), [0]))
        edges = np.flatnonzero(np.diff(padded)).reshape(-1, 2)
        for start, stop in edges:
            ax.axhspan(
                t_minutes[start], t_minutes[stop - 1],
                facecolor="none", edgecolor="red", hatch="//",
                linewidth=0, alpha=0.6, zorder=3,
            )
        if len(edges):
            ax.legend(
                handles=[Patch(facecolor="none", edgecolor="red", hatch="//",
                                label="in FOV")],
                fontsize=9, loc="upper right",
            )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel(f"Minutes since {Time(times[0], format='jd', scale='utc').iso}")
    ax.set_title(f"Waterfall: baseline ({ant1}, {ant2}), pol {pol}")
    return ax
