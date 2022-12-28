from pathlib import Path, PurePosixPath
from kedro.io import AbstractDataSet
import os
from tifffile import imread, imwrite
import numpy as np


class TiffDataSet(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = PurePosixPath(filepath)

    def _load(self) -> np.array:
        return imread(self._filepath)

    def _save(self, img: np.array) -> None:
        imwrite(str(self._filepath), img, compress=6)

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(name=os.path.basename(str(self._filepath)))
