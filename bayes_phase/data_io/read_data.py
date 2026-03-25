from pathlib import Path
from typing import Sequence

import polars as pl


def read_data(paths: Sequence[str | Path] | str | Path) -> pl.DataFrame:
    """Read a (list of) path(s) to .arrow files into a dataframe.

    Each file will be treated as an independent run of the same experiment, and should have identical columns and indices.

    Parameters
    ----------
    paths: Sequence[str | Path] | str | Path
        The (relative) paths to each .arrow file.

    Returns
    -------
    dataframe: polars.DataFrame
        The resulting dataframe, with a "run" column added that corresponds to the file each observation came from.
    """
    if isinstance(paths, Path) or isinstance(paths, str):
        paths = [paths]
    dfs = []
    for i, path in enumerate(paths):
        with open(path, "rb") as f:
            df = pl.read_ipc(f).with_columns(run=pl.lit(i + 1))
            dfs.append(df)
    return pl.concat(dfs)
