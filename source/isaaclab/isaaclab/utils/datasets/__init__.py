# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Submodule for datasets classes and methods.
"""

from .dataset_file_handler_base import DatasetFileHandlerBase
from .episode_data import EpisodeData

__all__ = ["DatasetFileHandlerBase", "EpisodeData", "HDF5DatasetFileHandler"]


def __getattr__(name: str):
    """Lazily import optional dataset handlers.

    HDF5 support depends on ``h5py`` and its native DLLs. Deferring the import
    keeps environments that do not export datasets from failing during startup.
    """

    if name == "HDF5DatasetFileHandler":
        from .hdf5_dataset_file_handler import HDF5DatasetFileHandler

        return HDF5DatasetFileHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
