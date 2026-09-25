"""
fetch_constellation_tles.py

Queries Space-Track.org for the archival TLE closest to a given date of
interest, for every satellite in a given constellation (matched by name,
e.g. STARLINK or ORBCOMM), and writes the result out in the 3-line
(name / line1 / line2) format that beam_vs_starlink.py's read_tle_file()
expects.

Requires a free Space-Track.org account:
    https://www.space-track.org/auth/createAccount
Set credentials as environment variables before running, 
or you'll be prompted for them interactively:
    export SPACETRACK_USER='you@example.com'
    export SPACETRACK_PASS='...'

Usage:
In day mode:
    python fetch_constellation_tles.py --constellation STARLINK \
        --date 2022-11-13 --out TLE2023_satellite_tle_busiest.txt
In season mode:
    python fetch_constellation_tles.py --constellation ORBCOMM \
        --season 2022 --outdir orbcomm_season_2022/tle/daily

Then in beam_vs_starlink.py:
    sats = read_tle_file("TLE2023_satellite_tle_busiest.txt")

Notes:
    - --date mode: only one TLE per satellite is kept, whichever has the
      EPOCH closest to --date, searched within +/- 3 days of
      --date. Active satellites typically get several TLEs per day.
    - --season YEAR mode: one archival TLE snapshot per satellite PER DAY
      across the whole season (Sept 1 YEAR to Apr 30 YEAR+1), written to
      --outdir as <date>_<constellation>_tles.txt - one file per day.
      Queries Space-Track one calendar month at a time because of query
      restrictions.
    - Debris fragments (OBJECT_NAME ending in " DEB") are skipped.
"""

import argparse
import bisect
import calendar
import getpass
import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta

from astropy.time import Time, TimeDelta
from astropy import units as u

LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
LOGOUT_URL = "https://www.space-track.org/ajaxauth/logout"
QUERY_URL_TMPL = (
    "https://www.space-track.org/basicspacedata/query/class/gp_history/"
    "OBJECT_NAME/~~{constellation}/EPOCH/{start}--{end}/"
    "orderby/NORAD_CAT_ID,EPOCH/format/json"
)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; hera-satellite-script/1.0)"}
# Space-Track rejects an EPOCH range this wide in one gp_history query
# ("Query range out of bounds") - --season mode queries one calendar month
# at a time instead of the whole season at once.
DATE_WINDOW_DAYS = 3.0  # --date mode: half-width of the EPOCH search window
REQUIRED_GP_FIELDS ={"OBJECT_NAME", "NORAD_CAT_ID", "EPOCH", "TLE_LINE1", "TLE_LINE2"}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--constellation", required=True,
        help="Constellation name to match against OBJECT_NAME, e.g. STARLINK "
             "or ORBCOMM (substring match).",
    )
    window = p.add_mutually_exclusive_group(required=True)
    window.add_argument(
        "--date",
        help="Date of interest, UTC, e.g. 2022-11-13 or 2022-11-13T00:10:00. "
             "One TLE per satellite, closest to this date, written to --out.",
    )
    window.add_argument(
        "--season", type=int, metavar="YEAR",
        help="Full observing season, Sept 1 of YEAR to Apr 30 of YEAR+1, "
             "e.g. --season 2022 for 2022-09-01 - 2023-04-30. One TLE file "
             "PER DAY, each satellite's own closest-to-that-day record, "
             "written to --outdir.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output TLE file path, 3-line format, --date mode only "
             "(default: <constellation>_tles.txt).",
    )
    p.add_argument(
        "--outdir", default=None,
        help="Output directory for one <date>_<constellation>_tles.txt "
             "per day, --season mode only (required with --season).",
    )
    args = p.parse_args()

    if args.date and args.outdir:
        p.error("--outdir is only valid with --season; use --out for --date mode.")
    if args.season and args.out:
        p.error("--out is only valid with --date mode; use --outdir for --season.")
    if args.season and not args.outdir:
        p.error("--season requires --outdir.")

    return args


def get_credentials():
    user = os.environ.get("SPACETRACK_USER") or input("Space-Track username/email: ")
    password = os.environ.get("SPACETRACK_PASS") or getpass.getpass("Space-Track password: ")
    return user, password


def make_session():
    """Build a urllib opener that persists the login cookie across requests."""
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def login(opener, user, password):
    data = urllib.parse.urlencode({"identity": user, "password": password}).encode()
    req = urllib.request.Request(LOGIN_URL, data=data, headers=HEADERS)
    try:
        with opener.open(req) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Space-Track login request failed: HTTP {exc.code}") from exc
    # Space-Track returns HTTP 200 even on bad credentials, with a JSON body
    # like {"Login":"Failed"} - so check the body, not just the status code.
    if "Failed" in body:
        raise RuntimeError(
            "Space-Track login rejected - check SPACETRACK_USER/SPACETRACK_PASS."
        )


def fetch_gp_history(opener, constellation, start, end):
    url = QUERY_URL_TMPL.format(
        constellation=urllib.parse.quote(constellation), start=start, end=end
    )
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with opener.open(req) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Space-Track query failed: HTTP {exc.code}") from exc
    return json.loads(body)


def validate_gp_records(records, label):
    """Exit with a clear message if records don't look like real GP history
    rows - Space-Track can return HTTP 200 with an error body instead of
    data (e.g. a query rejected for its EPOCH range), which would otherwise
    surface as a cryptic KeyError deep in later processing. `label`
    identifies which query this was, for the error message.
    """
    bad = [rec for rec in records if not REQUIRED_GP_FIELDS.issubset(rec)]
    if bad:
        sys.exit(
            f"Space-Track's response for {label} doesn't look like GP "
            f"history data - {len(bad)}/{len(records)} record(s) missing "
            f"expected fields. First such record:\n{bad[0]}"
        )


def is_debris(object_name):
    """True if OBJECT_NAME marks this as a debris fragment (e.g. "... DEB").

    Space-Track labels debris from a breakup with the parent satellite's
    name plus a " DEB" suffix, shared identically across every fragment.
    """
    return object_name.strip().upper().endswith(" DEB")


def closest_per_satellite(records, target_jd):
    """Keep only the record with EPOCH closest to target_jd for each NORAD ID."""
    best = {}
    for rec in records:
        norad = int(rec["NORAD_CAT_ID"])
        epoch_jd = Time(rec["EPOCH"], scale="utc").jd
        diff = abs(epoch_jd - target_jd)
        if norad not in best or diff < best[norad][0]:
            best[norad] = (diff, rec)
    return [rec for _, (diff, rec) in sorted(best.items())]


def write_tle_file(records, path):
    with open(path, "w") as fh:
        for rec in records:
            fh.write(f"{rec['OBJECT_NAME'].strip()}\n")
            fh.write(f"{rec['TLE_LINE1']}\n")
            fh.write(f"{rec['TLE_LINE2']}\n")


def run_season_mode(records, args):
    """Write one TLE file per day of the season from an already-fetched set
    of (debris-filtered) GP history records - pure record processing, no
    network I/O, so this can be exercised directly with fake records
    without a live Space-Track session.

    For each day from Sept 1 args.season to Apr 30 args.season+1 inclusive,
    picks each satellite's record with EPOCH closest to noon UTC that day -
    noon, not midnight, since HERA's night spans roughly 16:30-03:30 UTC
    (close to that night's midpoint), keeping this day label consistent
    with find_busy.dates_of()'s own dusk-labeled night convention. Space-
    Track won't necessarily have an observation for every satellite on
    every single day, so "closest available" can land a few days either
    side - the same fallback closest_per_satellite() already relies on for
    --date mode, just applied once per day here instead of once overall.

    Groups records by satellite with precomputed EPOCH Julian dates once,
    then bisects into each satellite's own sorted list per day, rather
    than re-scanning every record (as closest_per_satellite() does) once
    per day - this is what keeps the whole loop roughly
    O(records log records) instead of O(records * n_days).
    """
    by_norad = {}
    for rec in records:
        norad = int(rec["NORAD_CAT_ID"])
        epoch_jd = Time(rec["EPOCH"], scale="utc").jd
        by_norad.setdefault(norad, []).append((epoch_jd, rec))
    for pairs in by_norad.values():
        pairs.sort(key=lambda pair: pair[0])
    epoch_lists = {norad: [jd for jd, _ in pairs] for norad, pairs in by_norad.items()}

    os.makedirs(args.outdir, exist_ok=True)

    season_start = date(args.season, 9, 1)
    season_end = date(args.season + 1, 4, 30)
    n_days = (season_end - season_start).days + 1

    day = season_start
    n_written = 0
    while day <= season_end:
        target_jd = Time(f"{day}T12:00:00", scale="utc").jd
        day_records = []
        for norad, pairs in by_norad.items():
            jds = epoch_lists[norad]
            i = bisect.bisect_left(jds, target_jd)
            candidates = [c for c in (i - 1, i) if 0 <= c < len(jds)]
            best_i = min(candidates, key=lambda c: abs(jds[c] - target_jd))
            day_records.append(pairs[best_i][1])
        day_records.sort(key=lambda rec: int(rec["NORAD_CAT_ID"]))

        out_path = os.path.join(args.outdir, f"{day}_{args.constellation.lower()}_tles.txt")
        write_tle_file(day_records, out_path)
        n_written += 1
        if n_written % 20 == 0 or day == season_end:
            print(f"  ... {n_written}/{n_days} daily TLE files written (latest: {out_path})")
        day += timedelta(days=1)

    with open(os.path.join(args.outdir, "_SUCCESS"), "w") as fh:
        fh.write(f"{n_written} daily TLE files written for season "
                 f"{args.season}-{args.season + 1}\n")
    print(f"Wrote {n_written} daily TLE files to {args.outdir}")


def main():
    args = parse_args()

    user, password = get_credentials()
    opener = make_session()

    print(f"Logging in to Space-Track as {user}...")
    login(opener, user, password)

    if args.date:
        target = Time(args.date, scale="utc")
        window = TimeDelta(DATE_WINDOW_DAYS * u.day)
        start = (target - window).datetime.strftime("%Y-%m-%d")
        end = (target + window).datetime.strftime("%Y-%m-%d")

        print(f"Querying {args.constellation} GP history, EPOCH {start} to {end}...")
        records = fetch_gp_history(opener, args.constellation, start, end)
        print(f"Got {len(records)} TLE records.")
        if records:
            validate_gp_records(records, f"EPOCH {start} to {end}")
    else:
        # One query for the whole season hits Space-Track's EPOCH-range
        # limit ("Query range out of bounds") - query one calendar month
        # at a time instead (Sept args.season to Apr args.season+1).
        months = [(args.season, m) for m in range(9, 13)] + \
                 [(args.season + 1, m) for m in range(1, 5)]
        records = []
        for year, month in months:
            chunk_start = date(year, month, 1)
            chunk_end = date(year, month, calendar.monthrange(year, month)[1])
            print(f"Querying {args.constellation} GP history, EPOCH "
                  f"{chunk_start} to {chunk_end}...")
            chunk_records = fetch_gp_history(
                opener, args.constellation,
                chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d"),
            )
            if chunk_records:
                validate_gp_records(chunk_records, f"EPOCH {chunk_start} to {chunk_end}")
            records.extend(chunk_records)
        print(f"Got {len(records)} TLE records total across the season.")

    if not records:
        sys.exit(
            "No records returned - double-check the date/season, or check the --constellation spelling."
        )

    n_before = len(records)
    records = [rec for rec in records if not is_debris(rec["OBJECT_NAME"])]
    if n_before != len(records):
        print(f"Skipped {n_before - len(records)} debris fragments (name ending in "
              f"' DEB'), {len(records)} remaining.")

    if args.date:
        out_path = args.out or f"{args.constellation.lower()}_tles.txt"
        best = closest_per_satellite(records, target.jd)
        write_tle_file(best, out_path)
        print(f"Wrote {len(best)} satellites to {out_path}")
    else:
        run_season_mode(records, args)

    try:
        opener.open(urllib.request.Request(LOGOUT_URL, headers=HEADERS))
    except urllib.error.URLError:
        pass  # not critical if logout fails


if __name__ == "__main__":
    main()
