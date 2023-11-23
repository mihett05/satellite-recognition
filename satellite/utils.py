import math

# https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Tile_numbers_to_lon./lat.


def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 1 << zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile


def num2deg(xtile, ytile, zoom):
    n = 1 << zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return lat_deg, lon_deg


def location_to_pixel(
    lat: float,
    lon: float,
    bbox: tuple[float, float, float, float],  # left bottom right top
    width: int,
    height: int,
) -> tuple[int, int]:
    lat = math.radians(lat)
    lon = math.radians(lon)
    south, west, north, east = list(map(math.radians, bbox))
    ymin = mercator(south)
    ymax = mercator(north)
    x_factor = width / (east - west)
    y_factor = height / (ymax - ymin)

    y = mercator(lat)
    x = (lon - west) * x_factor
    y = (ymax - y) * y_factor
    return int(x), int(y)


def mercator(x: float) -> float:
    return math.log(math.tan(x / 2 + math.pi / 4))
