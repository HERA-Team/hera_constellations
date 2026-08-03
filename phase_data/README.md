# phase_data

Phases real HERA `UVData` to satellites as they cross the beam, building on
`find_satellites/run_pipeline.py`'s output (which finds every FOV crossing
for a given night/constellation and the busiest moment among them).

## Files

- **`phase_to_satellites.py`** - the reusable module. Key entry points:
  - `satellites_in_time_range(fov_dir, t_start, t_end)` - which satellites
    (from a `find_fov_passes.py` output directory) cross the FOV in a
    given window.
  - `phase_uvdata_to_satellite(uv, sat_name, tle_line1, tle_line2, ...)` -
    phase a copy of `uv` to one satellite's moving RA/Dec
    (`cat_type="ephem"`), restricted to the baseline-times it's in the
    beam.
  - `phase_uvdata_to_all_passes(uv, fov_dir, ...)` - do the above for every
    satellite crossing `uv`'s time range, returning `{name: phased UVData}`.
- **`example_phase_busiest_night.py`** - runnable script version of the
  demo below.
- **`phase_to_satellites_demo.ipynb`** - the same demo as a notebook, with
  explanations of *why* `cat_type="ephem"` is the right tool here, and the
  caveats below.

## Quick start

```python
from pyuvdata import UVData
import phase_to_satellites as pts

uv = UVData()
uv.read([...])  # your night's correlator files

fov_dir = "../find_satellites/<constellation>_<date>/fov_passes"  # from run_pipeline.py
phased = pts.phase_uvdata_to_all_passes(uv, fov_dir)
# {"STARLINK-1234": <UVData phased to that satellite, in-beam times only>, ...}
```

## Two things worth knowing before trusting the numbers

1. **Distance/parallax is deliberately dropped** when converting each
   satellite's AltAz position to the ICRS RA/Dec pyuvdata's `ephem` phase
   centers expect - only the direction is kept. Keeping the true range (a
   few hundred to ~2000 km) makes astropy apply diurnal parallax sized to
   Earth's *radius* (~6371 km, much bigger than the satellite's own
   range), which wildly overcorrects the result. This isn't a shortcut for
   convenience: pyuvdata's own `cat_type="ephem"` apparent-coordinate path
   doesn't apply a distance correction either (a `TODO` in
   `pyuvdata.utils.phasing` as of pyuvdata 3.2.1), so the plain topocentric
   apparent direction is exactly what it expects.
2. **Far-field (plane-wave) phasing is still fine for HERA's array itself**
   - the near-field defocus this ignores is of order B²/2D (baseline
   length squared over twice the range), a few cm of path length for
   HERA's baselines and a LEO satellite's range, far below the delay
   resolution of HERA's ~120 kHz channels.

See the notebook's last cell for the fuller writeup, including why
satellites overlapping in time need separate `UVData` copies rather than
one shared multi-phase-center object.

## No full night of real data here

This repo only has one ~10 s `UVData` snapshot
(`zen.2459876.44074.sum.uvh5`, at the repo root) checked in, which doesn't
overlap any satellite's FOV crossing. Both the script and notebook build a
synthetic `UVData` (real telescope/antenna metadata, made-up
zeroed visibilities) over the busiest window instead, purely to
demonstrate the phasing mechanics end to end - swap in `uv.read(<your
night's files>)` to run this for real.
