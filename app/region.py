"""Pick datacenters by distance. No user input, no configuration.

Distance dominates latency here: the same model measured 276 ms round-trip
on a European pod and 132-190 ms on a North American one. That 144 ms is
larger than anything buffer tuning recovers, so where the pod lands is the
single highest-value deployment decision -- and it is not something a user
should have to think about.

How it works, with no network call and no geolocation service:

  1. The OS timezone gives a longitude almost directly: UTC offset x 15
     degrees. That is exactly what timezones encode.
  2. The IANA prefix (America/, Asia/, Europe/) gives a usable latitude.
  3. Every Runpod datacenter has known coordinates, so we sort them by
     great-circle distance and hand Runpod the whole list in order.

Runpod takes the first with capacity, so a sold-out datacenter falls through
to the next-nearest rather than failing. Ranking every datacenter globally
means this works for a user anywhere without region-specific branches.

Datacenter ids verified against the Runpod console, September 2026.
"""

from __future__ import annotations

import math
import os
import time
from datetime import datetime

# id -> (latitude, longitude). Coordinates are the metro area; precision
# beyond that does not change the ordering.
#
# These ids come from the API's own dataCenterIds default in
# /v1/openapi.json -- NOT from the console UI. The console lists sites the
# API rejects (US-CO-1, US-MO-1, US-NE-1, CA-MTL-4, EUR-IS-4/5, EUR-NO-2),
# and a single invalid entry fails the whole create request with a schema
# error rather than naming the offender.
DATACENTERS: dict[str, tuple[float, float]] = {
    # North America
    "CA-MTL-1": (45.5, -73.6),
    "CA-MTL-2": (45.5, -73.6),
    "CA-MTL-3": (45.5, -73.6),
    "US-CA-2": (37.4, -122.1),
    "US-DE-1": (39.2, -75.5),
    "US-GA-1": (33.7, -84.4),
    "US-GA-2": (33.7, -84.4),
    "US-IL-1": (41.9, -87.6),
    "US-KS-2": (39.0, -95.7),
    "US-KS-3": (39.0, -95.7),
    "US-MD-1": (39.3, -76.6),
    "US-NC-1": (35.8, -78.6),
    "US-TX-1": (32.8, -96.8),
    "US-TX-3": (32.8, -96.8),
    "US-TX-4": (32.8, -96.8),
    "US-WA-1": (47.6, -122.3),
    # Europe
    "EU-CZ-1": (50.1, 14.4),
    "EU-FR-1": (48.9, 2.3),
    "EU-NL-1": (52.4, 4.9),
    "EU-RO-1": (44.4, 26.1),
    "EU-SE-1": (59.3, 18.1),
    "EUR-IS-1": (64.1, -21.9),
    "EUR-IS-2": (64.1, -21.9),
    "EUR-IS-3": (64.1, -21.9),
    "EUR-NO-1": (59.9, 10.7),
    # Asia / Oceania
    "AP-IN-1": (19.1, 72.9),
    "AP-JP-1": (35.7, 139.7),
    "OC-AU-1": (-33.9, 151.2),
}

# Known coordinates per IANA zone. Preferred over the UTC offset because
# the offset shifts with daylight saving and several zones share one -- both
# of which would move a user hundreds of km and reorder the ranking.
_ZONE_COORDS = {
    # US zones are continental in size -- America/Chicago runs from Texas to
    # Minnesota. Using the city would put a Texas user 1,400 km north, so
    # these are the CENTROID of each zone, which ranks fairly for everyone
    # inside it.
    "america/chicago": (37.5, -94.5), "america/new_york": (40.0, -77.5),
    "america/toronto": (44.5, -79.5), "america/montreal": (45.5, -73.6),
    "america/los_angeles": (38.0, -120.5), "america/vancouver": (49.3, -123.1),
    "america/denver": (40.0, -106.5), "america/phoenix": (33.4, -112.1),
    "america/detroit": (42.3, -83.0), "america/halifax": (44.6, -63.6),
    "america/winnipeg": (49.9, -97.1), "america/edmonton": (53.5, -113.5),
    "america/mexico_city": (19.4, -99.1), "america/bogota": (4.7, -74.1),
    "america/lima": (-12.0, -77.0), "america/sao_paulo": (-23.6, -46.6),
    "america/argentina/buenos_aires": (-34.6, -58.4),
    "europe/london": (51.5, -0.1), "europe/dublin": (53.3, -6.3),
    "europe/paris": (48.9, 2.3), "europe/berlin": (52.5, 13.4),
    "europe/amsterdam": (52.4, 4.9), "europe/madrid": (40.4, -3.7),
    "europe/rome": (41.9, 12.5), "europe/warsaw": (52.2, 21.0),
    "europe/prague": (50.1, 14.4), "europe/stockholm": (59.3, 18.1),
    "europe/oslo": (59.9, 10.7), "europe/bucharest": (44.4, 26.1),
    "europe/moscow": (55.8, 37.6), "europe/lisbon": (38.7, -9.1),
    "asia/kolkata": (19.1, 72.9), "asia/calcutta": (19.1, 72.9),
    "asia/karachi": (24.9, 67.0), "asia/colombo": (6.9, 79.9),
    "asia/dhaka": (23.8, 90.4), "asia/kathmandu": (27.7, 85.3),
    "asia/dubai": (25.2, 55.3), "asia/tehran": (35.7, 51.4),
    "asia/tokyo": (35.7, 139.7), "asia/seoul": (37.6, 127.0),
    "asia/shanghai": (31.2, 121.5), "asia/hong_kong": (22.3, 114.2),
    "asia/singapore": (1.4, 103.8), "asia/bangkok": (13.8, 100.5),
    "asia/jakarta": (-6.2, 106.8), "asia/manila": (14.6, 121.0),
    "australia/sydney": (-33.9, 151.2), "australia/melbourne": (-37.8, 145.0),
    "australia/perth": (-31.9, 115.9), "pacific/auckland": (-36.9, 174.8),
    "africa/lagos": (6.5, 3.4), "africa/johannesburg": (-26.2, 28.0),
    "africa/cairo": (30.0, 31.2), "africa/nairobi": (-1.3, 36.8),
}

# Fallback latitude per prefix, used with an offset-derived longitude when
# the exact zone is not in the table above.
_LATITUDE_HINTS = (
    ("america/argentina", -34.6), ("america/sao", -23.6),
    ("america/santiago", -33.4), ("america/lima", -12.0),
    ("america/bogota", 4.7), ("america/mexico", 19.4),
    ("america/", 39.0),
    ("europe/", 50.0),
    ("africa/", 6.5),
    ("asia/kolkata", 19.1), ("asia/calcutta", 19.1),
    ("asia/karachi", 24.9), ("asia/colombo", 6.9), ("asia/dhaka", 23.8),
    ("asia/dubai", 25.2), ("asia/tokyo", 35.7), ("asia/seoul", 37.6),
    ("asia/shanghai", 31.2), ("asia/singapore", 1.4),
    ("asia/", 28.0),
    ("australia/", -33.9), ("pacific/", -20.0),
    ("atlantic/", 38.0), ("indian/", -20.0),
)


def utc_offset_hours() -> float:
    off = datetime.now().astimezone().utcoffset()
    return off.total_seconds() / 3600.0 if off else 0.0


def timezone_name() -> str:
    """IANA name where the OS exposes it, else the abbreviation."""
    tz = os.getenv("TZ")
    if tz:
        return tz
    try:
        from pathlib import Path
        link = Path("/etc/localtime")
        if link.is_symlink():
            parts = str(link.resolve()).split("/zoneinfo/")
            if len(parts) == 2:
                return parts[1]
    except Exception:
        pass
    return time.tzname[0] if time.tzname else "?"


def estimate_location() -> tuple[float, float, str]:
    """Best guess at (lat, lon) plus a human explanation."""
    name = timezone_name()
    low = name.lower()

    exact = _ZONE_COORDS.get(low)
    if exact:
        return exact[0], exact[1], name

    lat = 39.0
    for prefix, value in _LATITUDE_HINTS:
        if low.startswith(prefix):
            lat = value
            break

    # Fallback: timezones are defined by longitude, so the offset is a fair
    # approximation when the zone itself is unknown.
    lon = max(-180.0, min(180.0, utc_offset_hours() * 15.0))
    return lat, lon, name


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in km."""
    r = 6371.0
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = p2 - p1
    dl = math.radians(b[1] - a[1])
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def continent_of(dc: str) -> str:
    if dc.startswith(("US-", "CA-")):
        return "NA"
    if dc.startswith(("EU-", "EUR-")):
        return "EU"
    if dc.startswith("OC-"):
        return "OC"
    return "AP"


def ranked_datacenters(limit: int | None = None) -> list[str]:
    """The whole nearest REGION, ordered by distance, then the rest.

    Naming a handful of specific datacenters fails outright when none of
    them has a free GPU -- measured: "could not find any pods with required
    specifications" for US-KS-2, and "not enough free GPUs on the host
    machine" when restarting. Capacity moves constantly.

    So we hand Runpod every datacenter on the user's continent, nearest
    first, and append the other continents as a last resort. Runpod takes
    the first with capacity, which means proximity is honoured when possible
    and the session still starts when it is not.
    """
    env = os.getenv("RUNPOD_DATACENTERS", "").strip()
    if env:
        return [d.strip() for d in env.split(",") if d.strip()]

    lat, lon, _ = estimate_location()
    here = (lat, lon)
    ordered = sorted(DATACENTERS.items(), key=lambda kv: haversine(here, kv[1]))

    home = continent_of(ordered[0][0])
    near = [dc for dc, _ in ordered if continent_of(dc) == home]
    far = [dc for dc, _ in ordered if continent_of(dc) != home]
    result = near + far
    return result[:limit] if limit else result


def home_region(limit: int | None = None) -> list[str]:
    """Only the nearest continent -- no transatlantic fallback."""
    all_dc = ranked_datacenters()
    home = continent_of(all_dc[0])
    near = [d for d in all_dc if continent_of(d) == home]
    return near[:limit] if limit else near


def describe() -> str:
    """One line for the UI: where we think you are and what we picked."""
    lat, lon, tz = estimate_location()
    near = home_region()
    if not near:
        return "no datacenter available"
    dist = haversine((lat, lon), DATACENTERS.get(near[0], (lat, lon)))
    return (f"{tz} -> {continent_of(near[0])} region, "
            f"{len(near)} datacenters, nearest {near[0]} (~{dist:,.0f} km)")


# Back-compat with earlier call sites.
def datacenters_for(_region_key: str | None = None) -> list[str]:
    return ranked_datacenters()
