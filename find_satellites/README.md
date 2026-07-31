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
[Space-Track.org](https://www.space-track.org/auth/createAccount) account.
Set credentials as environment variables before running (recommended, keeps
them out of shell history):

```bash
export SPACETRACK_USER='you@example.com'
export SPACETRACK_PASS='...'
```

If unset, you'll be prompted for them interactively instead.

## Modules

- **`run_pipeline.py`** - end-to-end pipeline for one constellation on one
  night: fetches TLEs, finds every FOV crossing, locates the busiest moment,
  and plots the trajectories around it. This is the main entry point - see
  the usage example below.
- **`fetch_constellation_tles.py`** - queries Space-Track for the archival
  TLE closest to a given date, for every satellite in a named constellation,
  and writes them out in the 3-line (name / line1 / line2) format the other
  scripts expect.
- **`find_fov_passes.py`** - batch FOV-pass finder: propagates every
  satellite in a TLE file with SGP4 (one worker process per satellite)
  across a night or a full observing season, restricted to night-time hours
  at HERA, and writes out every timestamp + position at which each one is
  inside the FOV. Produces one `<SATELLITE_NAME>_results.csv` per satellite
  plus a `Running_details.txt` with run parameters and resource usage.
- **`find_busy.py`** - reads `find_fov_passes.py`'s `*_results.csv` output
  and reports either the busiest/quietest calendar date across the
  observation window (`day` mode) or the busiest/quietest time-of-day across
  all nights (`time` mode).
- **`beam_vs_starlink.py`** - propagates satellites over a given time
  window, keeps only the positions inside HERA's beam, and plots their
  RA/Dec trajectories together with the beam's drifting outline. Can reuse
  `find_fov_passes.py`'s output (`--from-passes-dir`) to skip re-propagating
  satellites that never cross the beam in the window.
- **`parameters.py`** - shared physical/geometry constants (HERA's location,
  beam angular radius) used by `find_fov_passes.py` and
  `beam_vs_starlink.py`, so both stay in sync from a single source of truth.
- **`utils.py`** - shared functions used by the scripts above: TLE file
  parsing, building a time grid, filtering to night-time samples, the
  SGP4 -> TEME -> ITRS -> AltAz propagation loop, and matching satellites to
  a `find_fov_passes.py` output directory.
- **`requirements.txt`** - third-party Python packages needed to run
  everything above.

## Example: running the full pipeline

```bash
python run_pipeline.py --date 2022-11-13 --constellation Starlink
```

This runs, for the night starting on 2022-11-13 (dusk that day to dawn the
next):

1. Fetch TLEs for every Starlink satellite (`fetch_constellation_tles.py`).
2. Find every FOV crossing that night (`find_fov_passes.py`).
3. Find the busiest moment - the timestamp with the most satellites
   simultaneously in the beam (`find_busy.py`).
4. Plot the RA/Dec trajectories of every satellite in the beam during a
   window centred on that moment (`beam_vs_starlink.py`).

Useful optional flags:

```bash
python run_pipeline.py --date 2022-11-13 --constellation Orbcomm \
    --window-minutes 15 --outdir my_runs --no-progress
```

- `--window-minutes` - width of the trajectory-plot window (default: 10).
- `--busy-threshold` - fraction of satellites required for a moment to count
  as "busy" in `find_busy.py`'s own reference output (default: 0.90).
- `--limit` - cap the number of satellites propagated in the trajectory plot
  (step 4 only), for a quick test run.
- `--outdir` - base directory for outputs (default: `Pipeline_runs/` next to
  the script).
- `--no-progress` - disable the tqdm progress bars, useful for Slurm/batch
  jobs where in-place redraws just clutter the log file.

Outputs land under `<outdir>/<constellation>_<date>/`:

```
tle/<date>_<constellation>_tles.txt   - fetched TLEs (step 1)
fov_passes/<SAT>_results.csv          - per-satellite FOV crossings (step 2)
fov_passes/Running_details.txt        - run parameters/resource usage (step 2)
fov_passes/Time-satellite_correspondence.txt,
fov_passes/Busiest_time_positions.txt - find_busy.py's own reference outputs (step 3)
beam_busiest_window.png               - trajectory plot (step 4)
pipeline_summary.txt                  - busiest moment found, satellite count there,
                                         and paths to all of the above
```

Each script can also be run on its own - see the docstring at the top of
each file for its individual usage.
