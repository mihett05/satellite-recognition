import aiohttp


async def api_call(request: str) -> dict:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            data={"data": "[out:json];\n" + request + "\nout geom;"},
        ) as response:
            return (await response.json())["elements"]


async def get_building(bbox: tuple[float, float, float, float]):
    return await api_call(
        f"nwr[building][building!=house][building!=yes][building!=detached][building!=construction][building!=garage][building!=kiosk][building!=hut][building!=garages][building!=roof][building!=shed][location!=kiosk]({', '.join(map(str, bbox))});"
    )


async def get_roads(bbox: tuple[float, float, float, float]):
    return await api_call(f"way({', '.join(map(str, bbox))});")


async def get_buildings_and_roads(bbox: tuple[float, float, float, float]):
    return await api_call(
        """
(
  nwr[building]({{bbox}});
  way[highway]({{bbox}});
);
""".replace(
            "{{bbox}}", ", ".join(map(str, bbox))
        )
    )
