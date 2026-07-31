"""
parameters.py

Shared physical/geometry constants for the find_satellites/ scripts -
previously duplicated (byte-identical) between find_fov_passes.py and
beam_vs_starlink.py. Import by name (e.g. `from parameters import
hera_location, fov_angular_radius`), not via `import parameters` + a
`parameters.` prefix, so existing call sites elsewhere don't need renaming.
"""

import numpy as np
from astropy import units
from astropy.coordinates import EarthLocation

# HERA (Karoo, South Africa) - change accordingly for a different site.
hera_lat = -30.72152612068926 * units.deg
hera_lon = 21.42830382686301 * units.deg
hera_alt = 1051.69 * units.m

hera_location = EarthLocation.from_geodetic(lon=hera_lon, lat=hera_lat, height=hera_alt)

# Beam half-angle (radius) around HERA's zenith, so a 10 deg-wide beam.
fov_angular_radius = 5 * units.deg

# Plotting resolution, not a physical parameter: bearings swept around the
# beam center (0-360 deg) used to trace out its circular outline on the sky
# in beam_vs_starlink.py. Only used there currently.
position_angles = np.linspace(0, 360, 360) * units.deg
