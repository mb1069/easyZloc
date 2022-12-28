from pathlib import Path, PurePosixPath
from kedro.io import AbstractDataSet
import os
import numpy as np
import h5py

class SpotsDataSet(AbstractDataSet):
    def __init__(self, filepath, key):
        self._filepath = PurePosixPath(filepath)
        self.key = key

    def _load(self) -> np.array:
        with h5py.File(self._filepath, 'r') as f:
            return np.array(f[self.key])

    def _save(self, img: np.array) -> None:
        pass

    def _exists(self) -> bool:
        return Path(self._filepath.as_posix()).exists()

    def _describe(self):
        return dict(name=os.path.basename(str(self._filepath)))
