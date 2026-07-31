"""
fetch_constellation_tles.py

Queries Space-Track.org for the archival TLE closest to a given date of
interest, for every satellite in a given constellation (matched by name,
e.g. STARLINK or ORBCOMM), and writes the result out in the 3-line
(name / line1 / line2) format that beam_vs_starlink.py's read_tle_file()
expects.

Requires a free Space-Track.org account:
    https://www.space-track.org/auth/createAccount
Credentials are never hardcoded here - set them as environment variables
before running (recommended, keeps them out of shell history), or you'll
be prompted for them interactively:
    export SPACETRACK_USER='you@example.com'
    export SPACETRACK_PASS='...'

Usage:
    python fetch_constellation_tles.py --constellation STARLINK \
        --date 2022-11-13 --out TLE2023_satellite_tle_busiest.txt
    python fetch_constellation_tles.py --constellation ORBCOMM \
        --date 2022-11-13 --out orbcomm_tles.txt

Then in beam_vs_starlink.py:
    sats = read_tle_file("TLE2023_satellite_tle_busiest.txt")

Notes:
    - Only one TLE per satellite is kept: whichever has the EPOCH closest
      to --date. --window-days controls how far Space-Track is searched
      around --date for candidates; widen it if a satellite you expect
      ends up missing (it just means it wasn't tracked within the window).
    - The constellation filter is a simple "name contains --constellation"
      match (e.g. ORBCOMM satellites are typically named "ORBCOMM FM ..."),
      so it could in principle also pick up an oddly-named unrelated
      object; spot-check the output if that matters for your use case.
"""

import argparse
import getpass
import http.cookiejar
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

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


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--constellation", required=True,
        help="Constellation name to match against OBJECT_NAME, e.g. STARLINK "
             "or ORBCOMM (substring match).",
    )
    p.add_argument(
        "--date", required=True,
        help="Date of interest, UTC, e.g. 2022-11-13 or 2022-11-13T00:10:00",
    )
    p.add_argument(
        "--window-days", type=float, default=3.0,
        help="Half-width, in days, of the EPOCH search window around --date "
             "(default: 3).",
    )
    p.add_argument(
        "--out", default=None,
        help="Output TLE file path, 3-line format "
             "(default: <constellation>_tles.txt)",
    )
    return p.parse_args()


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


def main():
    args = parse_args()
    out_path = args.out or f"{args.constellation.lower()}_tles.txt"

    target = Time(args.date, scale="utc")
    window = TimeDelta(args.window_days * u.day)
    start = (target - window).datetime.strftime("%Y-%m-%d")
    end = (target + window).datetime.strftime("%Y-%m-%d")

    user, password = get_credentials()
    opener = make_session()

    print(f"Logging in to Space-Track as {user}...")
    login(opener, user, password)

    print(f"Querying {args.constellation} GP history, EPOCH {start} to {end}...")
    records = fetch_gp_history(opener, args.constellation, start, end)
    print(f"Got {len(records)} TLE records.")

    if not records:
        sys.exit(
            "No records returned - try widening --window-days, double-check the "
            "date, or check the --constellation spelling."
        )

    best = closest_per_satellite(records, target.jd)
    write_tle_file(best, out_path)
    print(f"Wrote {len(best)} satellites to {out_path}")

    try:
        opener.open(urllib.request.Request(LOGOUT_URL, headers=HEADERS))
    except urllib.error.URLError:
        pass  # not critical if logout fails


if __name__ == "__main__":
    main()
