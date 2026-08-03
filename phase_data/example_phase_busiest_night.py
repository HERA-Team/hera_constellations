"""
example_phase_busiest_night.py

End-to-end example: take run_pipeline.py's output for a given night, find
the busiest moment, and phase real HERA telescope metadata (antenna
layout, location) to the satellite(s) crossing the beam around that
moment - using phase_to_satellites.py's cat_type="ephem" approach.

This repo doesn't currently have a full night of real correlator data
committed (only one ~10 s snapshot, zen.2459876.44074.sum.uvh5, which
falls outside every satellite pass that night - see step 2 below), so
this example builds a *synthetic* UVData object for the demonstration: it
reuses the real telescope (antenna positions, location) read from that
snapshot file, but fabricates a time_array spanning the busiest window and
`data_array` filled with dummy noise. Swap step 3's UVData.new() call for
a plain uv.read(<your files for this night>) to run this on real data -
everything after that point is unchanged.

Usage:
    python example_phase_busiest_night.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy import units
from astropy.time import Time
from pyuvdata import UVData

import phase_to_satellites as pts
from parameters import fov_angular_radius, position_angles

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# An already-computed run_pipeline.py output (find_satellites/run_pipeline.py
# --date 2022-10-23 --constellation Orbcomm). Point this at a Starlink run's
# directory instead (after fetching Starlink TLEs with a Space-Track.org
# account - see find_satellites/README.md) to reproduce this for Starlink.
PIPELINE_RUN_DIR = REPO_ROOT / "find_satellites" / "orbcomm_2022-10-23"
FOV_DIR = PIPELINE_RUN_DIR / "fov_passes"

SAMPLE_UVH5 = REPO_ROOT / "zen.2459876.44074.sum.uvh5"


def main():
    # --- Step 1: find the busiest moment from run_pipeline.py's output ------
    summary = (PIPELINE_RUN_DIR / "pipeline_summary.txt").read_text()
    print("=== run_pipeline.py summary for this night ===")
    print(summary)

    # --- Step 2: real telescope metadata, from whatever local UVData file --
    # exists (only used for its antenna layout/location - Ntimes here is 2,
    # spanning ~10 s around 22:34:35 UTC on this night, which doesn't overlap
    # any satellite's FOV crossing per FOV_DIR's CSVs).
    uv_real = UVData()
    uv_real.read(str(SAMPLE_UVH5), read_data=False)
    telescope = uv_real.telescope
    print(
        f"\nReal telescope metadata from {SAMPLE_UVH5.name}: "
        f"{telescope.Nants} antennas, location {telescope.location}"
    )

    real_start = Time(uv_real.time_array.min(), format="jd")
    real_end = Time(uv_real.time_array.max(), format="jd")
    no_overlap = pts.satellites_in_time_range(FOV_DIR, real_start, real_end)
    print(
        f"Satellites crossing the FOV during this file's own time range "
        f"({real_start.iso} to {real_end.iso}): {list(no_overlap) or 'none'}"
    )

    # --- Step 3: synthetic UVData over the busiest window --------------------
    # A handful of real antennas/baselines, one representative frequency
    # channel (~137.5 MHz, in Orbcomm's downlink band) and a 5 s cadence
    # (matching find_fov_passes.py's --day sampling, chosen for the same
    # reason: fine enough not to step over a ~20 s FOV crossing).
    antnums = telescope.antenna_numbers[:4]
    antpairs = [(a, b) for i, a in enumerate(antnums) for b in antnums[i + 1:]]

    t0 = Time("2022-10-23T21:08:00", scale="utc")
    times = t0 + np.arange(0, 271, 5) * units.s

    freq_idx = int(np.argmin(np.abs(uv_real.freq_array - 137.5e6)))
    freq_array = uv_real.freq_array[freq_idx: freq_idx + 1]

    uv = UVData.new(
        freq_array=freq_array,
        polarization_array=uv_real.polarization_array[:1],
        times=times.jd,
        telescope=telescope,
        antpairs=antpairs,
        do_blt_outer=True,
        integration_time=5.0,
        channel_width=uv_real.channel_width[freq_idx],
        empty=True,  # data_array/flag_array/nsample_array all zeroed - a
                     # stand-in for real correlator output over this window.
    )
    print(
        f"\nSynthetic UVData: {uv.Ntimes} times x {uv.Nbls} baselines x "
        f"{uv.Nfreqs} freq channel, {times[0].iso} to {times[-1].iso}"
    )

    # --- Step 4: phase to every satellite crossing this window --------------
    print("\n=== Phasing to satellite passes in this window ===")
    phased = pts.phase_uvdata_to_all_passes(uv, FOV_DIR)

    if not phased:
        print("No satellite passes in this window - nothing to plot.")
        return

    # --- Step 5: sanity-check plot -------------------------------------------
    # For each satellite, the phase center's recovered apparent RA/Dec
    # track (from uv_sat.phase_center_app_ra/dec, i.e. what pyuvdata will
    # actually use to compute uvw) should sit right on top of HERA's FOV
    # disk at zenith at each corresponding time - confirming the ephemeris
    # was interpolated/applied correctly, the same style of plot
    # find_satellites/beam_vs_starlink.py and test_orbcomm_14.py use.
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    for sat_name, uv_sat in phased.items():
        unique_t, idx = np.unique(uv_sat.time_array, return_index=True)
        ra = np.degrees(uv_sat.phase_center_app_ra[idx])
        dec = np.degrees(uv_sat.phase_center_app_dec[idx])
        ax.plot(ra, dec, "o-", label=sat_name)

        # FOV disk outline at the moment closest to the middle of the pass.
        mid = Time(unique_t[len(unique_t) // 2], format="jd", scale="utc")
        from astropy.coordinates import AltAz, ICRS, SkyCoord
        from parameters import hera_location
        zenith = SkyCoord(
            AltAz(
                alt=90 * units.deg, az=0 * units.deg, obstime=mid,
                location=hera_location,
            )
        ).transform_to(ICRS())
        ring = zenith.directional_offset_by(position_angles, fov_angular_radius)
        ax.fill(
            ring.ra.wrap_at(360 * units.deg).deg, ring.dec.deg,
            alpha=0.15, color="C0",
            label="HERA FOV (at pass midpoint)" if sat_name == next(iter(phased)) else None,
        )

    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title("Ephem phase center track vs. HERA's FOV")
    ax.legend(fontsize=8)
    outfile = SCRIPT_DIR / "phase_center_vs_fov.png"
    fig.savefig(outfile, dpi=150)
    print(f"\nSaved sanity-check plot to {outfile}")


if __name__ == "__main__":
    main()
