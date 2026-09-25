# Satellite constellations in HERA data

Tools for studying how low-Earth-orbit satellite constellations (Starlink,
Orbcomm) show up in data from the Hydrogen Epoch of Reionization Array
(HERA). HERA sits in the Karoo, South Africa. It is a drift-scan telescope
with a narrow beam around zenith, so a satellite contaminates the data only
while it is crossing that beam.

The repo has two main parts, which you run in order:

1. **[`find_satellites/`](find_satellites/)** works out *when* satellites are
   in HERA's field of view.
2. **[`phase_data/`](phase_data/)** uses those crossings to *phase HERA
   visibilities* to each satellite as it moves through the beam.

## `find_satellites/`: when are satellites in the beam?

For a given constellation and a given night (or a whole observing season),
this package:

- fetches archival TLEs (orbital elements) for every satellite in the
  constellation from [Space-Track.org](https://www.space-track.org);
- propagates each satellite with SGP4 across night-time hours at HERA and
  records every timestamp at which it is inside the field of view. The field
  of view is approximated as a disk of radius 5° centred on zenith;
- finds the busiest moment, meaning the time with the most satellites in the
  beam at once, or the busiest night across a season;
- plots the RA/Dec trajectories of every satellite in the beam around that
  moment, together with the beam's drifting outline.

The main entry point runs all four steps for one night:

```bash
cd find_satellites
python run_pipeline.py --day 2022-11-13 --constellation Starlink
```

Outputs are written to `find_satellites/Pipeline_runs/<constellation>_<date>/`.
The per-satellite FOV-crossing tables in its `fov_passes/` subfolder are what
`phase_data/` reads. See [`find_satellites/README.md`](find_satellites/README.md)
for setup (including Space-Track credentials), the individual scripts, and
all options.

## `phase_data/`: phasing HERA visibilities to satellites

HERA's raw data are phased to zenith, not to any fixed sky position. To look
for a satellite's signal, you need to re-phase the visibilities so that they
track that satellite as it moves across the sky.

For every satellite crossing the time range of a `UVData` object,
[`phase_to_satellites.py`](phase_data/phase_to_satellites.py):

- re-propagates the satellite's TLE at the data's own timestamps to get its
  RA/Dec track;
- phases a copy of the data to that moving position using pyuvdata's
  `cat_type="ephem"` phase centers;
- keeps only the baseline-times when the satellite was actually in the beam.

The result is one phased `UVData` object per satellite:

```python
from pyuvdata import UVData
import phase_to_satellites as pts

uv = UVData()
uv.read([...])  # HERA data for the night

fov_dir = "../find_satellites/Pipeline_runs/<constellation>_<date>/fov_passes"
phased = pts.phase_uvdata_to_all_passes(uv, fov_dir)
# {"STARLINK-1234": <UVData phased to that satellite, in-beam times only>, ...}
```

A satellite's position is converted to RA/Dec using its direction only, and
its distance is dropped on purpose. Phasing is far-field, which is accurate
enough for HERA's baselines at LEO distances. The module docstring and
[`phase_data/README.md`](phase_data/README.md) explain both choices. For a
worked example, see [`example_phase_busiest_night.py`](phase_data/example_phase_busiest_night.py)
or the demo notebook
[`phase_to_satellites_demo.ipynb`](phase_data/phase_to_satellites_demo.ipynb).

`phase_data/` imports `parameters.py` and `utils.py` from `find_satellites/`,
so keep the two folders side by side.

## Requirements

- `find_satellites/`: `pip install -r find_satellites/requirements.txt`
  (astropy, matplotlib, numpy, psutil, sgp4, tqdm), plus a free Space-Track
  account to fetch TLEs.
- `phase_data/`: the packages above plus [pyuvdata](https://pyuvdata.readthedocs.io).

## Other files

- `tles/`: TLE files that were already fetched for Starlink and Orbcomm on a
  few dates.
- `satellite_positions.ipynb`, `antenna_positions.ipynb`, `read_data.py`,
  `plot_waterfalls.py`: exploratory notebooks and scripts for looking at
  satellite positions, the HERA array layout, and visibility waterfalls.

## License

MIT. See [LICENSE](LICENSE).
