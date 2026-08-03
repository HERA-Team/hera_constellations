"""
phase_to_satellites.py

Phase HERA UVData visibilities to Starlink/Orbcomm satellites as they cross
the field of view, reusing the per-satellite FOV-crossing tables that
find_satellites/run_pipeline.py (specifically its find_fov_passes.py step)
already computed for a given night.

Why "ephem" and not a single phase() call
------------------------------------------
HERA is a drift-scan telescope: its raw/'sum' UVData files are phased to
zenith (an "unprojected" phase center) rather than to any fixed sky
position.

pyuvdata's answer is the "ephem" phase center type:
    UVData.phase(cat_name=..., cat_type="ephem", ephem_times=<JD array>,
                 ra=<rad array>, dec=<rad array>, phase_frame="icrs")
(https://pyuvdata.readthedocs.io/en/latest/uvdata.html#pyuvdata.UVData.phase)
You supply the source's ICRS RA/Dec at a handful of times and pyuvdata
interpolates the apparent position to every timestamp being phased.

This module builds those RA/Dec(t) ephemerides from the same TLEs
run_pipeline.py already fetched and re-propagates them with SGP4 at the 
UVData file's own timestamps, via find_satellites/utils.py's
propagate_in_beam().

Far-field vs near-field phasing
--------------------------------
UVData.phase(cat_type="ephem") applies standard far-field (plane-wave)
phasing: every antenna is assumed to see the source along parallel rays.
For a LEO satellite (~600-2000 km altitude) and HERA's few-hundred-metre
baselines B, the near-field defocus this ignores is of order B^2 / (2 D)
in path length (D = range to satellite) - e.g. B=300 m, D=700 km gives
~6 cm, i.e. ~0.2 ns of delay error, far below HERA's ~120 kHz channel
resolution (~8 microsecond delay resolution). Far-field phasing is
adequate here; pyuvdata's "near_field" cat_type exists for cases where it
isn't (only accepts a single fixed lon/lat/dist per call, so tracking a
moving near-field source with it means calling phase() once per
timestamp).

One phase center per satellite, not one shared object
--------------------------------------------------------
A UVData object's phase_center_id_array assigns exactly one phase center
to each baseline-time - so two satellites simultaneously in the beam can't
both be the "current" phase center of the same object at the same time.
phase_uvdata_to_all_passes() below sidesteps this by returning a separate
UVData copy per satellite, each containing only the baseline-times that
satellite was actually in the FOV - directly usable for per-satellite
analysis (e.g. plotting the recovered phase center track, or - with real
correlator data - looking for a slowly-varying/coherent visibility at that
phase center, the signature of a correctly-phased RFI source).
"""

import sys
from pathlib import Path

import numpy as np
from astropy import units
from astropy.coordinates import AltAz, ICRS
from astropy.time import Time

# find_satellites/ lives alongside this folder (both under the repo root),
# not on sys.path by default - add it so the imports below work regardless
# of the caller's working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
FIND_SATELLITES_DIR = REPO_ROOT / "find_satellites"
if str(FIND_SATELLITES_DIR) not in sys.path:
    sys.path.insert(0, str(FIND_SATELLITES_DIR))

from parameters import hera_location, fov_angular_radius  # noqa: E402
from utils import propagate_in_beam  # noqa: E402


def satellites_in_time_range(fov_dir, t_start, t_end):
    """Every satellite with >=1 find_fov_passes.py row inside [t_start, t_end].

    Returns {satellite_name: (tle_line1, tle_line2)}. This is a coarse 
    pre-filter using find_fov_passes.py's fixed-cadence CSVs; 
    phase_uvdata_to_satellite() below does the precise re-propagation, 
    at the UVData file's own timestamps.

    t_start/t_end: anything astropy.time.Time accepts.
    fov_dir: a find_fov_passes.py output directory (e.g.
        find_satellites/<constellation>_<date>/fov_passes).
    """
    start_iso, end_iso = Time(t_start).iso, Time(t_end).iso
    found = {}
    for csv_path in sorted(Path(fov_dir).glob("*_results.csv")):
        name = line1 = line2 = None
        hit = False
        with open(csv_path) as fh:
            for line in fh:
                if line.startswith("# satellite:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("# tle_line1:"):
                    line1 = line.split(":", 1)[1].strip()
                elif line.startswith("# tle_line2:"):
                    line2 = line.split(":", 1)[1].strip()
                elif line.startswith("#") or line.startswith("timestamp,"):
                    continue
                else:
                    ts = line.split(",", 1)[0]
                    if start_iso <= ts <= end_iso:
                        hit = True
                        break
        if hit and name and line1 and line2:
            found[name] = (line1, line2)
    return found


def satellite_ephemeris(tle_line1, tle_line2, times, location=hera_location,
                         fov_radius=fov_angular_radius, in_beam_only=True):
    """Re-propagate one satellite's TLE at exactly `times`, in ICRS RA/Dec.

    times: an astropy Time array (typically a UVData file's own unique
    timestamps) to propagate at.

    Returns `None` if the satellite is never inside `fov_radius` of zenith
    at any of `times`. Otherwise a dict with:
        'idx'    - positions of the returned samples within `times`
        'time'   - the corresponding Time values
        'ra','dec' - ICRS position at each sample (Quantity, deg)
        'alt','az','zensep' - topocentric position at each sample (Quantity, deg),
                   as in find_fov_passes.py's own output columns

    Thin wrapper around find_satellites/utils.py's propagate_in_beam() -
    the same SGP4 -> TEME -> ITRS -> AltAz propagation find_fov_passes.py
    used to generate the CSVs in the first place - plus an AltAz -> ICRS
    transform, since pyuvdata's phase_frame="icrs" (the default) expects
    RA/Dec in that frame.
    """
    idxs, out_times = [], []
    ras, decs, alts, azs, zenseps = [], [], [], [], []
    for idx, t1, _itrs, altaz, zensep in propagate_in_beam(
        "", tle_line1, tle_line2, times, location, fov_radius,
        in_beam_only=in_beam_only,
    ):
        # Deliberately drop the satellite's true (few-hundred-to-2000 km)
        # range here, keeping only its alt/az *direction* - i.e. treat it
        # as an infinite-distance celestial source, since:
        #   1. Far-field array phasing itself assumes a source at infinity,
        #      keeping the AltAz frame's real `altaz.distance` here would
        #      make astropy apply diurnal parallax appropriate to Earth's
        #      radius (~6371 km), which is large compared to the
        #      satellite's actual range and wildly overcorrects: the RA/Dec
        #      it produces barely moves at all across a pass, even though
        #      the true topocentric alt/az sweeps by degrees per second.
        #   2. pyuvdata's own apparent-coordinate pipeline for cat_type=
        #      "ephem" does not apply a distance/parallax correction. 
        direction = AltAz(
            alt=altaz.alt, az=altaz.az, obstime=t1, location=location
        )
        icrs = direction.transform_to(ICRS())
        idxs.append(idx)
        out_times.append(t1)
        ras.append(icrs.ra.to_value(units.deg))
        decs.append(icrs.dec.to_value(units.deg))
        alts.append(altaz.alt.to_value(units.deg))
        azs.append(altaz.az.to_value(units.deg))
        zenseps.append(zensep.to_value(units.deg))

    if not out_times:
        return None

    return {
        "idx": np.array(idxs, dtype=int),
        "time": Time(out_times),
        "ra": np.array(ras) * units.deg,
        "dec": np.array(decs) * units.deg,
        "alt": np.array(alts) * units.deg,
        "az": np.array(azs) * units.deg,
        "zensep": np.array(zenseps) * units.deg,
    }


def phase_uvdata_to_satellite(uv, sat_name, tle_line1, tle_line2,
                               in_beam_only=True, location=hera_location,
                               fov_radius=fov_angular_radius):
    """Return a copy of `uv`, phased to one satellite's moving RA/Dec.

    The copy is first select()ed down to the baseline-times `sat_name` was
    inside HERA's FOV (or, if in_beam_only=False, to `uv`'s full time
    range), then phase()d to it with cat_type="ephem".

    Returns `None` if the satellite is never in the beam during any of
    `uv`'s own timestamps (only possible with in_beam_only=True) e.g.
    because the coarse filter in satellites_in_time_range() only matched
    find_fov_passes.py's coarser sampling grid nearby, not `uv`'s exact
    timestamps.
    """
    unique_jd = np.unique(uv.time_array)
    unique_times = Time(unique_jd, format="jd", scale="utc")

    ephem = satellite_ephemeris(
        tle_line1, tle_line2, unique_times, location=location,
        fov_radius=fov_radius, in_beam_only=in_beam_only,
    )
    if ephem is None:
        return None

    # ephem['idx'] indexes into unique_times/unique_jd, slicing unique_jd, 
    # rather than round-tripping ephem['time'] back through .jd, keeps 
    # this an exact float comparison, immune to any Time jd1/jd2 rounding.
    in_pass_jd = unique_jd[ephem["idx"]]
    select_mask = np.isin(uv.time_array, in_pass_jd)
    if not select_mask.any():
        return None

    uv_sat = uv.select(blt_inds=np.nonzero(select_mask)[0], inplace=False)
    uv_sat.phase(
        cat_name=sat_name,
        cat_type="ephem",
        ephem_times=ephem["time"].jd,
        ra=ephem["ra"].to_value(units.rad),
        dec=ephem["dec"].to_value(units.rad),
        phase_frame="icrs",
        epoch="J2000",
    )
    return uv_sat


def phase_uvdata_to_all_passes(uv, fov_dir, satellites=None, in_beam_only=True,
                                verbose=True):
    """Phase `uv` to every satellite that crosses the FOV during its time range.

    fov_dir: a find_fov_passes.py output directory for the night in
        question (e.g. find_satellites/<constellation>_<date>/fov_passes,
        as written by run_pipeline.py).
    satellites: optional iterable of satellite names (matching each CSV's
        '# satellite:' header) to restrict to; default checks every
        satellite with a results CSV in fov_dir.
    in_beam_only: passed through to phase_uvdata_to_satellite() - if
        False, each returned copy spans `uv`'s *entire* time range phased
        to that satellite (e.g. to inspect far-sidelobe contamination
        before/after the main FOV transit), rather than just the in-beam
        samples.

    Returns {satellite_name: phased UVData}, one entry per satellite that
    actually overlaps `uv`'s own timestamps (in_beam_only=True skips any
    satellite that doesn't, even if find_fov_passes.py's coarser grid
    nominally caught it nearby).
    """
    t_start = Time(uv.time_array.min(), format="jd", scale="utc")
    t_end = Time(uv.time_array.max(), format="jd", scale="utc")
    candidates = satellites_in_time_range(fov_dir, t_start, t_end)
    if satellites is not None:
        wanted = set(satellites)
        candidates = {k: v for k, v in candidates.items() if k in wanted}

    results = {}
    for sat_name, (line1, line2) in candidates.items():
        uv_sat = phase_uvdata_to_satellite(
            uv, sat_name, line1, line2, in_beam_only=in_beam_only,
        )
        if uv_sat is not None:
            results[sat_name] = uv_sat
            if verbose:
                print(
                    f"Phased to {sat_name}: {uv_sat.Ntimes} time sample(s), "
                    f"{Time(uv_sat.time_array.min(), format='jd').iso} to "
                    f"{Time(uv_sat.time_array.max(), format='jd').iso}"
                )
        elif verbose:
            print(
                f"{sat_name}: nearby in find_fov_passes.py's output, but no "
                f"overlap with this file's exact timestamps "
                f"(in_beam_only={in_beam_only})."
            )
    return results
