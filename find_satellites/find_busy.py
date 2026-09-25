"""
find_busy.py

Combines find_busy_day.py and find_busy_time_point.py: reads every
"*_results.csv" file in the current directory (produced by
"Check FOV Passes/find_fov_passes.py") and reports either the
busiest/quietest observing night across the observation window, or the
busiest/quietest TIME-OF-DAY (HH:MM:SS, on the script's sampling grid)
across all nights.

A night/time counts as "busy" if at least `--threshold` (default 90%) of
all tracked satellites had an FOV pass on it.

"day" mode labels each night by the UTC date it starts on: dusk to dawn.

Once the busiest night/time is identified, the script also writes one
satellite-timestamp-position line per crossing (for the winning night, or
every occurrence of the winning time-of-day) to a text file. Positions
(Alt/Az/zenith separation) are read directly from each satellite's CSV.

Usage:
    python find_busy.py day   [--threshold 0.90]
    python find_busy.py time  [--threshold 0.90]
"""

import argparse
import csv
import glob

from astropy.time import Time
from astropy import units as u

from utils import night_label


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["day", "time"],
        help="'day': busiest/quietest observing night, labeled by the UTC "
             "date it starts on (dusk that day to dawn the next). "
             "'time': busiest/quietest time-of-day across all nights.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.90,
        help="Fraction of satellites required for a night/time to count as "
             "'busy' (default: 0.90, i.e. 90%%).",
    )
    return parser.parse_args()


def read_sat_passes():
    """Read every "*_results.csv" file into satellite_name -> list of passes.

    Each results file (written by find_fov_passes.py::work_func_sat)
    has a few '#'-prefixed metadata comment lines, a CSV header row, then one
    row per FOV crossing: timestamp,alt_deg,az_deg,zensep_deg.

    Returns sat_passes: satellite_name -> list of
    (timestamp_str, alt_deg, az_deg, zensep_deg) tuples. A satellite with no
    FOV crossings simply maps to an empty list.
    """
    files = sorted(glob.glob("*_results.csv"))
    if not files:
        raise SystemExit("No '*_results.csv' files found in the current directory.")

    sat_passes = {}
    for path in files:
        name = path.rsplit("/", 1)[-1]                  # e.g. "STARLINK-1234_results.csv"
        sat_name = name.split("_results", 1)[0]          # e.g. "STARLINK-1234"

        with open(path, encoding="utf-8", newline="") as f:
            rows = [row for row in csv.reader(f) if row and not row[0].startswith("#")]

        if not rows:
            print(f"Warning: {path} has no header row (truncated file?); skipping.")
            continue

        # rows[0] is the "timestamp,alt_deg,az_deg,zensep_deg" header row.
        sat_passes[sat_name] = [
            (timestamp_str, float(alt_str), float(az_str), float(zensep_str))
            for timestamp_str, alt_str, az_str, zensep_str in rows[1:]
        ]

    return sat_passes


def read_time_delta_s():
    """Read the time_delta_s recorded by find_fov_passes.py in any one
    *_results.csv's header comment, from the current directory (every
    file from the same run carries the same value). Returns None if no
    CSV has that comment (e.g. an older file that predates it).
    """
    for path in sorted(glob.glob("*_results.csv")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# time_delta_s:"):
                    try:
                        return float(line.split(":", 1)[1].strip())
                    except ValueError:
                        return None
                if not line.startswith("#"):
                    break
    return None


def collapse_near_duplicates(timestamps, time_delta_s, min_gap_multiples=5):
    """Collapse one satellite's own timestamps into one per FOV crossing.

    A single continuous pass gets sampled at several consecutive
    time_delta steps (e.g. 17:13:25, 17:13:30, 17:13:35), which would
    otherwise count as several separate "busiest moments" for what's
    really one event. Keeps the first timestamp of each cluster of
    timestamps whose consecutive gaps are all under
    min_gap_multiples * time_delta_s. If time_delta_s is None (couldn't
    be determined), returns the input unchanged.
    """
    if time_delta_s is None:
        return set(timestamps)
    ts_sorted = sorted(Time(t, scale="utc") for t in timestamps)
    if not ts_sorted:
        return set()
    min_gap = min_gap_multiples * time_delta_s
    kept = [ts_sorted[0]]
    for prev, t in zip(ts_sorted, ts_sorted[1:]):
        if (t - prev).sec >= min_gap:
            kept.append(t)
    return {t.iso for t in kept}


def dates_of(sat_passes):
    """satellite -> set of night labels ("YYYY-MM-DD") it had an FOV pass on.

    A night is labeled by the UTC date it starts on - dusk that day to dawn
    the next. utils.night_label() shifts each timestamp
    back 12h before taking its date so passes just before and just after
    UTC midnight.
    """
    return {
        sat: {night_label(Time(ts, scale="utc")) for ts, *_ in passes}
        for sat, passes in sat_passes.items()
    }


def times_of(sat_passes):
    """satellite -> set of times-of-day ("HH:MM:SS.mmm") it passed at."""
    return {sat: {ts.split()[1] for ts, *_ in passes} for sat, passes in sat_passes.items()}


def compute_pass_counts(sat_pass_map):
    """Given satellite -> set of keys (dates or times), count satellites per key."""
    all_keys = set().union(*sat_pass_map.values())
    pass_counts = {}
    for key in all_keys:
        count = 0
        for keys in sat_pass_map.values():
            if key in keys:
                count += 1
        pass_counts[key] = count
    return pass_counts


def top_n_busiest(pass_counts, n=5):
    """Return the n busiest keys from a key -> count map.

    A list of (key, count) tuples, sorted by count descending then by key
    ascending for ties (chronological order for date/timestamp keys).
    Works for any pass_counts map regardless of what the keys represent -
    night labels (dates_of()), full timestamps, or times-of-day (times_of()).
    """
    return sorted(pass_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def busy_sparse_split(pass_counts, threshold_fraction, total_sats):
    """Split keys into >=threshold ("busy") and <threshold ("sparse") sets."""
    min_required = threshold_fraction * total_sats
    busy = sorted(key for key, count in pass_counts.items() if count >= min_required)
    sparse = sorted(key for key, count in pass_counts.items() if count < min_required)
    return busy, sparse, min_required


def extreme_split(pass_counts):
    """Return (busiest_keys, busiest_count, quietest_keys, quietest_count)."""
    max_count = max(pass_counts.values())
    min_count = min(pass_counts.values())
    busiest = sorted(key for key, count in pass_counts.items() if count == max_count)
    quietest = sorted(key for key, count in pass_counts.items() if count == min_count)
    return busiest, max_count, quietest, min_count


def satellites_at(sat_pass_map, keys):
    """Every satellite that has at least one of `keys` in its pass set."""
    sats = []
    for sat_name, sat_keys in sat_pass_map.items():
        if sat_keys.intersection(keys):
            sats.append(sat_name)
    return sorted(sats)


def write_positions_file(filename, header_line, entries):
    """Write one line per (satellite, timestamp, alt, az, zensep).

    entries: iterable of (satellite_name, timestamp_str, alt_deg, az_deg,
    zensep_deg) tuples, in the order they should be written.
    """
    print(f"Satellite positions at busiest time(s) printed to {filename}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header_line + "\n\n")
        f.write(f"{'Satellite':<20}{'Timestamp (UTC)':<26}{'Alt (deg)':>12}{'Az (deg)':>12}{'ZenSep (deg)':>14}\n")
        for sat_name, timestamp_str, alt, az, zensep in entries:
            f.write(f"{sat_name:<20}{timestamp_str:<26}{alt:>12.4f}{az:>12.4f}{zensep:>14.4f}\n")


# --------------------------------- Modes ------------------------------------

def run_day_mode(sat_passes, threshold_fraction):
    sat_pass_dates = dates_of(sat_passes)
    total_sats = len(sat_pass_dates)
    pass_counts = compute_pass_counts(sat_pass_dates)
    busy_dates, sparse_dates, min_required = busy_sparse_split(
        pass_counts, threshold_fraction, total_sats)
    busiest_dates, max_count, quietest_dates, min_count = extreme_split(pass_counts)
    max_date = busiest_dates[0]
    min_date = quietest_dates[0]
    pct_label = f"{threshold_fraction * 100:.0f}%"

    # Satellites missing on each below-threshold date
    misses_by_date = {}
    for date in sparse_dates:
        missed_sats = [sat for sat, dates in sat_pass_dates.items() if date not in dates]
        misses_by_date[date] = missed_sats

    file_output_1 = "Dates_with_missing_pass.txt"
    print(f"Dates with missing pass printed to {file_output_1}")
    with open(file_output_1, "w", encoding="utf-8") as f:
        f.write(f"Busy threshold: >= {pct_label} of satellites ({min_required:.1f}/{total_sats})\n\n")
        f.write(f"Dates with >= {pct_label} of satellites passing: \n{busy_dates} \n\n")
        f.write(f"Dates below {pct_label} of satellites passing: \n{sparse_dates} \n\n")

    file_output_2 = "Date-satellite_missing.txt"
    with open(file_output_2, "w", encoding="utf-8") as f:
        print(f"Date and corresponding missing satellite printed to {file_output_2}")
        f.write("SUMMARY:\n")
        f.write(f"  Busy threshold: >= {pct_label} of satellites ({min_required:.1f}/{total_sats})\n")
        f.write(f"  Most passes on {max_date}: {max_count}/{total_sats}\n")
        f.write(f"  Fewest passes on {min_date}: {min_count}/{total_sats}\n\n\n")

        for date, missing in misses_by_date.items():
            f.write(f"{date} -> missing: {', '.join(missing)}\n")

    # Satellite/timestamp/position breakdown for the busiest date.
    entries = sorted(
        (sat, ts, alt, az, zensep)
        for sat, passes in sat_passes.items()
        for ts, alt, az, zensep in passes
        if ts.split()[0] == max_date
    )
    write_positions_file(
        "Busiest_day_positions.txt",
        f"Satellites in HERA's FOV on the busiest date ({max_date}, {max_count}/{total_sats} satellites):",
        entries,
    )

    print("\nSUMMARY:")
    print(f"Most passes on {max_date}: {max_count}/{total_sats}")
    print(f"Fewest passes on {min_date}: {min_count}/{total_sats}")


def run_time_mode(sat_passes, threshold_fraction):
    sat_pass_times = times_of(sat_passes)
    total_sats = len(sat_pass_times)
    pass_counts = compute_pass_counts(sat_pass_times)
    busy_times, sparse_times, min_required = busy_sparse_split(
        pass_counts, threshold_fraction, total_sats)
    busiest_times, busiest_count, quietest_times, quietest_count = extreme_split(pass_counts)
    pct_label = f"{threshold_fraction * 100:.0f}%"

    sats_at_busiest = satellites_at(sat_pass_times, busiest_times)
    sats_at_quietest = satellites_at(sat_pass_times, quietest_times)

    output_time_file = "Time-satellite_correspondence.txt"
    with open(output_time_file, "w", encoding="utf-8") as f:
        print(f"Time and corresponding satellite printed to {output_time_file}")
        f.write("SUMMARY:\n")
        f.write(f"  Busy threshold: >= {pct_label} of satellites ({min_required:.1f}/{total_sats})\n")
        f.write(f"  Times with >= {pct_label} of satellites passing: \n{busy_times} \n\n")
        f.write(f"  Times below {pct_label} of satellites passing: \n{sparse_times} \n\n")
        f.write(f"  Busiest time: {busiest_times}\nwith {busiest_count}/{total_sats} satellites passing\n\n")
        f.write(f"  Quietest time: {quietest_times}\nwith {quietest_count}/{total_sats} satellites passing\n\n")

        f.write(f"Satellites at busiest time ({busiest_times}):\n")
        for sat in sats_at_busiest:
            f.write(f"{sat}\n")
        f.write("\n")

        f.write(f"Satellites at quietest time ({quietest_times}):\n")
        for sat in sats_at_quietest:
            f.write(f"{sat}\n")

    # Satellite/timestamp/position breakdown for every occurrence (any date)
    # of the busiest time-of-day slot(s).
    busiest_times_set = set(busiest_times)
    entries = sorted(
        (sat, ts, alt, az, zensep)
        for sat, passes in sat_passes.items()
        for ts, alt, az, zensep in passes
        if ts.split()[1] in busiest_times_set
    )
    write_positions_file(
        "Busiest_time_positions.txt",
        f"Satellites in HERA's FOV at the busiest time-of-day {busiest_times} "
        f"({busiest_count}/{total_sats} satellites, across all nights):",
        entries,
    )

    print("\nSUMMARY:")
    print(f"Busiest time: {busiest_times}\nwith {busiest_count}/{total_sats} satellites passing")
    print(f"Quietest time: {quietest_times}\nwith {quietest_count}/{total_sats} satellites passing")


if __name__ == "__main__":
    args = parse_args()
    sat_passes = read_sat_passes()

    if args.mode == "day":
        run_day_mode(sat_passes, args.threshold)
    else:
        run_time_mode(sat_passes, args.threshold)
