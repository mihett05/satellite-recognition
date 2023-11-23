import asyncio
from shutil import which
from aiohttp import ClientSession
from .utils import deg2num

URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"


async def download_tile(x: int, y: int, z: int) -> bytes:
    async with ClientSession(
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0"
        }
    ) as session:
        async with session.get(URL.format(x=x, y=y, z=z)) as response:
            return await response.read()


async def download_tile_with_shift(
    x: int, y: int, z: int, vshift: int, hshift: int
) -> tuple[bytes, tuple[int, int]]:
    return (await download_tile(x + vshift, y + hshift, z), (x, y))


async def download_bbox_tiles(
    latlon_lb: tuple[float, float], latlon_rt: tuple[float, float], zoom: int
) -> list[list[bytes]]:
    left, bottom = deg2num(*latlon_lb, zoom)
    right, top = deg2num(*latlon_rt, zoom)
    tiles = [(x, y, zoom) for x in range(right - left) for y in range(0, bottom - top)]
    print(f"Downloading {len(tiles)} tiles")
    result = await asyncio.gather(
        *[download_tile_with_shift(*tile, vshift=left, hshift=top) for tile in tiles]
    )
    grid = [[b"" for x in range(right - left)] for y in range(0, bottom - top)]
    for b, (x, y) in result:
        grid[y][x] = b
    return grid
