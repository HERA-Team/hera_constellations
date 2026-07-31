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

from astropy import units
from astropy.timeseries import TimeSeries
# TEME: satellite-native True Equator Mean Equinox frame SGP4 propagates in.
# ITRS: Earth-fixed frame used as a stepping stone to the topocentric view.
# AltAz + get_sun: used both for the night-time filter and the FOV test.
from astropy.coordinates import (
    TEME, CartesianDifferential, CartesianRepresentation, ITRS, AltAz, get_sun,
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
    `times` (any iterable of astropy Time objects - a plain array or a
    tqdm-wrapped one, this function doesn't care) runs the shared physics
    chain: SGP4 -> TEME -> ITRS -> AltAz, testing whether the result falls
    within `fov_radius` of `location`'s zenith. SGP4 errors are always
    silently skipped, matching the two scripts' prior behavior.

    in_beam_only: if True (default), samples outside `fov_radius` of zenith
    are skipped too, so only true beam crossings are yielded - this is what
    find_fov_passes.py and beam_vs_starlink.py's default mode rely on. Pass
    False to yield every sample regardless of beam membership (zen_sep is
    still computed and returned either way) - used by beam_vs_starlink.py's
    --all-positions mode to reproduce the old rafas.py behaviour of plotting
    a satellite's whole track, for comparison against the beam-filtered view.

    Note: no tqdm or multiprocessing-position logic here on purpose - that's
    specific to how each caller runs inside a Pool worker (see their own
    work_func_sat), and doesn't belong in an otherwise plain, independently
    testable generator. Wrap `times` in tqdm(...) before passing it in if a
    progress bar is wanted; enumerate() below works the same either way.

    Yields (idx, t1, itrs, altaz, zen_sep) - idx is the position of t1 in
    the original `times` sequence (needed by callers that write into a
    pre-sized array indexed by sample position), itrs/altaz are the
    corresponding coordinate frame objects (so callers can pull out
    whatever they specifically need - geodetic position, RA/Dec via a
    further transform, etc.), and zen_sep is the angular distance from
    zenith already computed once, rather than recomputed by every caller.
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

        # ITRS -> AltAz as seen from `location` specifically (not Earth's
        # center): a LEO satellite's altitude is a large fraction of
        # Earth's radius, so its apparent position depends strongly on the
        # observer's location (topocentric parallax).
        altaz = itrs.transform_to(AltAz(obstime=t1, location=location))

        # Angular distance from zenith; inside the FOV disk if within
        # fov_radius of it.
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
