import os
import shutil
from glob import glob
from pathlib import Path


def unite_datasets(
    names: list[str],
    result_name: str,
    filter: bool = False,
    convert_yolo: bool = True,
):
    base = Path("datasets") / result_name
    os.mkdir(base)

    if not convert_yolo:
        os.mkdir(base / "images")
        os.mkdir(base / "masks")

    counter = 0
    for name in names:
        if convert_yolo:
            for file in glob(str(Path("datasets") / name / "**" / "*.png")):
                p = Path(file)
                if (
                    filter
                    and os.path.getsize(p.parent / (p.name.split(".")[0] + ".txt")) < 0
                ):
                    continue
                shutil.move(p, base / f"{counter:04}.png")
                shutil.move(
                    p.parent / (p.name.split(".")[0] + ".txt"),
                    base / f"{counter:04}.txt",
                )
                counter += 1
        else:
            for file in glob(str(Path("datasets") / name / "images" / "*.png")):
                p = Path(file)
                shutil.move(p, base / "images" / f"{counter:04}.png")
                shutil.move(
                    p.parent.parent / "masks" / p.name,
                    base / "masks" / f"{counter:04}.png",
                )
                counter += 1
        shutil.rmtree(Path("datasets") / name)
