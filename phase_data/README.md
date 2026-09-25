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
- **`phase_to_satellites_demo.ipynb`** - the same demo as a notebook.

## Quick start

```python
from pyuvdata import UVData
import phase_to_satellites as pts

uv = UVData()
uv.read([...])  # your night's data

fov_dir = "../find_satellites/Pipeline_runs/<constellation>_<date>/fov_passes"  # from run_pipeline.py
phased = pts.phase_uvdata_to_all_passes(uv, fov_dir)
# {"STARLINK-1234": <UVData phased to that satellite, in-beam times only>, ...}
```

## Two things worth knowing before trusting the numbers

1. **Distance/parallax is deliberately dropped** when converting each
   satellite's AltAz position to the ICRS RA/Dec pyuvdata's `ephem` phase
   centers expect and only the direction is kept. Keeping the true range (a few hundred to couple thousands km) makes `astropy` apply diurnal parallax sized to Earth's radius, which wildly overcorrects the result.
2. **Far-field phasing** is used. `UVData.phase(cat_type="ephem")` applies
   standard far-field (plane-wave) phasing: every antenna is assumed to see
   the source along parallel rays, which is a fair approximation given a LEO satellite and HERA's layout.

