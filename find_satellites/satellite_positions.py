"""
satellite_positions.py

Alt/az of every satellite in a set of TLEs, at any Julian Date(s), as plain
importable functions for a notebook or another module (unlike
find_fov_passes.py, which parses command-line arguments at import time):

    import sys
    sys.path.insert(0, "<path to hera_constellations>/find_satellites")
    import satellite_positions as sp

    jd = 2459897.3   # 2022-11-13 19:12 UTC
    tles = sp.fetch_tles("STARLINK", jd, cache_dir="tle_cache")  # or sp.read_tles(path)
    pos = sp.satellite_altaz(tles, jd)
    pos.names[pos.alt > 85]    # satellites within 5 deg of zenith

SGP4 runs once over all satellites and times together (sgp4's SatrecArray),
and the TEME -> ITRS -> AltAz conversion is applied to the whole
(Nsat, Ntimes) array in one astropy call, rather than one call per satellite
per timestamp as in utils.propagate_in_beam().
"""

import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date
from os import PathLike
from pathlib import Path

import numpy as np
from astropy import units
from astropy.coordinates import TEME, ITRS, AltAz, CartesianRepresentation
from astropy.time import Time
from sgp4.api import Satrec, SatrecArray

import fetch_constellation_tles as fct
from parameters import hera_location
from utils import read_tle_file, night_label


def _as_time(jd):
    """UTC Julian Date(s), or an astropy Time, as a UTC astropy Time."""
    if isinstance(jd, Time):
        return jd.utc
    return Time(np.asarray(jd, dtype=float), format="jd", scale="utc")


def _daily_tle_file(directory, night):
    """The <YYYY-MM-DD>_<constellation>_tles.txt file in `directory` for
    `night` (a 'YYYY-MM-DD' string), or for the nearest date present if
    that night's file is missing, the same fallback as
    utils.propagate_in_beam_daily_tle().
    """
    by_date = {}
    for path in Path(directory).glob("*_tles.txt"):
        match = re.match(r"^(\d{4}-\d{2}-\d{2})_", path.name)
        if match:
            by_date[match.group(1)] = path
    if not by_date:
        raise FileNotFoundError(
            f"No <YYYY-MM-DD>_<constellation>_tles.txt files in {directory}."
        )
    if night not in by_date:
        target = date.fromisoformat(night)
        # sorted() so a tie goes to the earlier date, as it does there.
        night = min(sorted(by_date), key=lambda d: abs((date.fromisoformat(d) - target).days))
    return by_date[night]


def read_tles(source, jd=None):
    """Read TLEs from disk, as a list of (name, line1, line2).

    source: either a TLE file in the 3-line format (name, line 1, line 2),
        as written by fetch_constellation_tles.py --date or by fetch_tles();
        or a directory of daily <YYYY-MM-DD>_<constellation>_tles.txt files,
        as written by fetch_constellation_tles.py --season.
    jd: only used, and then required, when `source` is a directory. It
        picks the file for the night containing jd (noon-to-noon UTC, the
        same convention as find_fov_passes.py --day), or the nearest date
        available if that night's file is missing. All of jd must fall in
        one night.
    """
    source = Path(source)
    if not source.is_dir():
        return read_tle_file(source)

    if jd is None:
        raise ValueError(f"{source} is a directory of daily TLEs: pass jd to pick a day.")
    t = _as_time(jd)
    night = night_label(t.min())
    if night_label(t.max()) != night:
        raise ValueError(
            "jd spans more than one night, and each night has its own daily "
            "TLE file. Call once per night."
        )
    return read_tle_file(_daily_tle_file(source, night))


def fetch_tles(constellation, jd, cache_dir=None, window_days=3.0):
    """Fetch, from Space-Track, each satellite's TLE with epoch closest to jd.

    Runs the same query and filtering as fetch_constellation_tles.py --date
    (debris dropped, closest epoch within +/- window_days of jd). Needs a
    Space-Track account: set SPACETRACK_USER and SPACETRACK_PASS in the
    environment, or you'll be prompted for them.

    constellation: substring matched against Space-Track's OBJECT_NAME,
        e.g. "STARLINK" or "ORBCOMM".
    jd: UTC Julian Date, or an astropy Time. For an array, its median is
        used.
    cache_dir: if given, the TLEs are written there as
        <constellation>_JD<jd>_tles.txt, and later calls with the same
        constellation and jd read that file instead of querying Space-Track
        again.

    Returns a list of (name, line1, line2), like read_tles().
    """
    t = _as_time(jd)
    if not t.isscalar:
        t = Time(np.median(t.jd), format="jd", scale="utc")

    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"{constellation.lower()}_JD{t.jd:.5f}_tles.txt"
        if cache_file.exists():
            return read_tle_file(cache_file)

    start = (t - window_days * units.day).strftime("%Y-%m-%d")
    end = (t + window_days * units.day).strftime("%Y-%m-%d")
    opener = fct.make_session()
    fct.login(opener, *fct.get_credentials())
    try:
        records = fct.fetch_gp_history(opener, constellation, start, end)
    finally:
        try:
            opener.open(urllib.request.Request(fct.LOGOUT_URL, headers=fct.HEADERS))
        except urllib.error.URLError:
            pass  # not critical if logout fails

    # Space-Track can answer HTTP 200 with an error body instead of data.
    if not isinstance(records, list) or not all(
        isinstance(rec, dict) and fct.REQUIRED_GP_FIELDS.issubset(rec) for rec in records
    ):
        raise RuntimeError(
            f"Space-Track's response doesn't look like GP history data: {str(records)[:500]}"
        )
    records = [rec for rec in records if not fct.is_debris(rec["OBJECT_NAME"])]
    if not records:
        raise RuntimeError(
            f"No {constellation} TLEs with epochs from {start} to {end}. Check "
            "the constellation's spelling, or widen window_days."
        )

    best = fct.closest_per_satellite(records, t.jd)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        fct.write_tle_file(best, cache_file)
    return [(rec["OBJECT_NAME"].strip(), rec["TLE_LINE1"], rec["TLE_LINE2"]) for rec in best]


@dataclass
class SatellitePositions:
    """Topocentric positions of a set of satellites at one or more times.

    alt, az and dist have shape (Nsat,) when satellite_altaz() was given a
    single JD, or (Nsat, Ntimes) for an array of JDs. They are NaN wherever
    SGP4 failed, e.g. after a satellite decayed.
    """
    names: np.ndarray         # (Nsat,) satellite names, from the TLEs
    jd: np.ndarray            # scalar or (Ntimes,) UTC Julian Date(s)
    alt: np.ndarray           # deg above the horizon
    az: np.ndarray            # deg, east of north
    dist: np.ndarray          # km from the observing site
    tle_epoch_jd: np.ndarray  # (Nsat,) UTC JD of each TLE's epoch


def satellite_altaz(tles, jd, location=hera_location):
    """Alt/az of every satellite in `tles` at `jd`, as seen from `location`.

    tles: a list of (name, line1, line2), as returned by read_tles() or
        fetch_tles(); a {name: (line1, line2)} dict, as returned by
        phase_to_satellites.satellites_in_time_range(); or a path, which is
        passed to read_tles() along with jd.
    jd: UTC Julian Date(s): a scalar, an array, or an astropy Time.
    location: observing site, as an astropy EarthLocation. Defaults to HERA.

    Returns a SatellitePositions. Positions are only as good as the TLEs,
    which go out of date within days of their epoch, so compare jd against
    the returned tle_epoch_jd.
    """
    if isinstance(tles, (str, PathLike)):
        tles = read_tles(tles, jd)
    elif isinstance(tles, dict):
        tles = [(name, line1, line2) for name, (line1, line2) in tles.items()]
    if len(tles) == 0:
        raise ValueError("No TLEs given.")

    names = np.array([name.strip() for name, _, _ in tles])
    sats = [Satrec.twoline2rv(line1, line2) for _, line1, line2 in tles]

    t = _as_time(jd)
    scalar = t.isscalar
    t = t.reshape(-1)

    # SGP4 on every satellite at every time in one call. Positions come out
    # in the TEME frame, in km, with shape (Nsat, Ntimes, 3). A nonzero
    # error code (e.g. a satellite that has already decayed) becomes NaN.
    err, r_teme, _ = SatrecArray(sats).sgp4(t.jd1, t.jd2)
    r_teme[err != 0] = np.nan

    # TEME -> ITRS (Earth-fixed). obstime has shape (Ntimes,), which
    # broadcasts against the (Nsat, Ntimes) positions.
    teme = TEME(CartesianRepresentation(np.moveaxis(r_teme, -1, 0) * units.km), obstime=t)
    itrs = teme.transform_to(ITRS(obstime=t))

    # Geocentric -> topocentric, by subtracting the site's own ITRS
    # position, then rotate into the local horizon frame. This is the
    # purely geometric route from astropy's satellite docs
    # (https://docs.astropy.org/en/stable/coordinates/satellites.html).
    # Transforming geocentric ITRS straight to AltAz gives nearly the same
    # answer but detours through CIRS, which is much slower.
    topo = itrs.cartesian - location.get_itrs(t).cartesian
    altaz = ITRS(topo, obstime=t, location=location).transform_to(
        AltAz(obstime=t, location=location)
    )

    def per_sat(x):
        return x[:, 0] if scalar else x

    return SatellitePositions(
        names=names,
        jd=t.jd[0] if scalar else t.jd,
        alt=per_sat(altaz.alt.deg),
        az=per_sat(altaz.az.deg),
        dist=per_sat(altaz.distance.to_value(units.km)),
        tle_epoch_jd=np.array([sat.jdsatepoch + sat.jdsatepochF for sat in sats]),
    )
