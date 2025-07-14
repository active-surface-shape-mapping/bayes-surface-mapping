from pathlib import Path
from typing import Sequence

import polars as pl


def read_data(paths: Sequence[str | Path] | str | Path) -> pl.DataFrame:
    if isinstance(paths, Path) or isinstance(paths, str):
        paths = [paths]
    dfs = []
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            df = pl.read_ipc(f).with_columns(run=pl.lit(i + 1))
            dfs.append(df)
    return pl.concat(dfs)


if __name__ == "__main__":
    print(
        read_data(
            [
                "data/111_Cv_run1.arrow",
                "data/111_Cv_run2.arrow",
                "data/111_Cv_run3.arrow",
            ]
        ).filter(pl.col("temp_in_K") == 10.0)
    )
