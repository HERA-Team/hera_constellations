# find_satellites

Finds when satellite constellations (Starlink, Orbcomm) pass through HERA's
field of view (approximated to be a 5deg radius disk centred on zenith) on a given night or over a whole season, identifies the moment with the most
satellites simultaneously in the beam, and plots their sky trajectories
during that window.

## Setup

```bash
pip install -r requirements.txt
```

Fetching TLEs (`fetch_constellation_tles.py`) also requires a free
[space-Track.org](https://www.space-track.org/auth/createAccount) account.
Set credentials as environment variables before running (recommended, keeps
them out of shell history):

```bash
export SPACETRACK_USER='you@example.com'
export SPACETRACK_PASS='...'
```

If unset, you'll be prompted for them interactively instead.

## Modules

- **`run_pipeline.py`**: end-to-end pipeline for one constellation on one
  night: fetches TLEs, finds every FOV crossing, locates the busiest moment,
  and plots the trajectories around it. This is the main entry point - see
  the usage example below.
- **`fetch_constellation_tles.py`** queries Space-Track for the archival
  TLE closest to a given date, for every satellite in a named constellation,
  and writes them out in the 3-line (name / line1 / line2) format the other
  scripts expect.
- **`find_fov_passes.py`** is a batch FOV-pass finder: propagates every
  satellite in a TLE file with SGP4 across a night or a full observing season, 
  restricted to night-time hours at the HERA location,
  and writes out every timestamp + position at which each one is
  inside the FOV. Produces one `<SATELLITE_NAME>_results.csv` per satellite
  plus a `Running_details.txt` with run parameters and resource usage.
- **`find_busy.py`** reads `find_fov_passes.py`'s `*_results.csv` output
  and reports either the busiest/quietest calendar date across the
  observation window (`day` mode) or the busiest/quietest time-of-day across
  all nights (`time` mode).
- **`beam_vs_starlink.py`** propagates satellites over a given time
  window, keeps only the positions inside HERA's beam, and plots their
  RA/Dec trajectories together with the beam's drifting outline. Can reuse
  `find_fov_passes.py`'s output (`--from-passes-dir`) to skip re-propagating
  satellites that never cross the beam in the window.
- **`parameters.py`** holds shared physical/geometry constants.
- **`utils.py`** holds shared functions such as TLE file
  parsing, building a time grid, filtering to night-time samples, the
  SGP4 -> TEME -> ITRS -> AltAz propagation loop.

## Example: running the full pipeline

`run_pipeline.py` runs in one of two modes: `--day` for a single night, or
`--season` for a whole observing season. You must pass exactly one of them.

### One night (`--day`)

```bash
python run_pipeline.py --day 2022-11-13 --constellation Starlink
```

This runs, for the night starting on 2022-11-13 (dusk that day to dawn the
next):

1. Fetch TLEs for every Starlink satellite: one archival snapshot per
   satellite, the one closest to that date (`fetch_constellation_tles.py`).
2. Find every FOV crossing that night (`find_fov_passes.py`).
3. Find the busiest moment, ie the timestamp with the most satellites
   simultaneously in the beam (`find_busy.py`).
4. Plot the RA/Dec trajectories of every satellite in the beam during a
   window centred on that moment (`beam_vs_starlink.py`).

### A full season (`--season`)

```bash
python run_pipeline.py --season 2022 --constellation Orbcomm
```

`--season YEAR` covers HERA's observing season, from 1 September of `YEAR`
to 30 April of `YEAR+1` (here 2022-09-01 to 2023-04-30). The steps are the
same, with three differences:

1. TLEs are fetched once per day in the season to ensure the TLE used to 
   predict the satellite's trajectory is accurate.
2. Instead of the busiest moment, the pipeline finds the busiest night. \
   A night is labelled by the UTC date it starts on.
3. The trajectory plot covers that whole night, dusk to dawn.

A season run fetches and propagates far more data than a single night, so
it takes much longer. For Orbcomm in 2022, expect a couple hours.

### Optional flags

```bash
python run_pipeline.py --day 2022-11-13 --constellation Orbcomm \
    --window-minutes 15 --outdir my_runs --no-progress
```

- `--window-minutes`: width of the trajectory-plot window (default: 10 minutes).
  Can only be used with the `--day` option.
- `--top-n`: how many of the busiest moments (`--day`) or nights
  (`--season`) to list in `pipeline_summary.txt` (default: 5).
- `--limit` caps the number of satellites propagated in the trajectory plot
  (step 4 only), for a quick test run.
- `--outdir` specifies the base directory for outputs (default: `Pipeline_runs/` next to
  the script).
- `--no-progress` disables the progress bars.

### Outputs

Outputs land under `<outdir>/<constellation>_<label>/`, where `<label>` is
the date for `--day` (e.g. `Pipeline_runs/orbcomm_2022-11-13/`) or
`season_<YEAR>` for `--season` (e.g. `Pipeline_runs/orbcomm_season_2022/`):

| File | Description |
| --- | --- |
| `tle/<date>_<constellation>_tles.txt` | Fetched TLEs (step 1). `--day` only. |
| `tle/daily/<date>_<constellation>_tles.txt` | One TLE file per day of the season (step 1). `--season` only. |
| `fov_passes/<SATNAME>_results.csv` | Per-satellite FOV crossings (step 2). |
| `fov_passes/Running_details.txt` | Run parameters and resource usage (step 2). |
| `beam_busiest_window_<date>.png` | Trajectory plot (step 4). `<date>` is the `--day` date, or the busiest night for `--season`. |
| `beam_busiest_window_<date>_fov_times.txt` | Entry and exit times of each satellite in the trajectory plot (step 4). |
| `pipeline_summary.txt` | Busiest moment or night found and its satellite count, the `--top-n` busiest moments or nights with which satellites were in the beam, and paths to all of the above. |

The `fov_passes/` folder is what [`../phase_data/`](../phase_data/) reads.

For more detailed statistics (busy and quiet dates, which satellites never
passed on a given night, satellite positions at the busiest time), run
`find_busy.py` on a run's `fov_passes/` folder.

To list the busiest moments or nights again from an existing run, without
re-running the pipeline, use `find_busiest.py`:

```bash
python find_busiest.py Pipeline_runs/orbcomm_season_2022_1/fov_passes --season --top 10
python find_busiest.py Pipeline_runs/orbcomm_2022-11-15/fov_passes --day 2022-11-15 --top 5
```

Each script can also be run on its own - see the docstring at the top of
each file for its individual usage.
