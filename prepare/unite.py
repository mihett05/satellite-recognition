import os
import shutil
from glob import glob
from pathlib import Path


def unite_datasets(names: list[str], result_name: str):
    base = Path("datasets") / result_name
    os.mkdir(base)

    counter = 0
    for name in names:
        for file in glob(str(Path("datasets") / name / "**" / "*.png")):
            p = Path(file)
            shutil.move(p, base / f"{counter:04}.png")
            shutil.move(
                p.parent / (p.name.split(".")[0] + ".txt"),
                base / f"{counter:04}.txt",
            )
            counter += 1
