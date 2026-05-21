import pickle
import sys

import numpy.core
import numpy.core.multiarray
import numpy.core.numeric


class _CompatUnpickler(pickle.Unpickler):
    """Load legacy SB3 normalization stats pickled with NumPy 2 module paths."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = "numpy.core" + module[len("numpy._core") :]
        return super().find_class(module, name)


def load_vec_normalize(load_path, venv):
    with open(load_path, "rb") as file_handler:
        vec_normalize = _CompatUnpickler(file_handler).load()
    vec_normalize.set_venv(venv)
    return vec_normalize


def install_numpy_compat_aliases():
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
