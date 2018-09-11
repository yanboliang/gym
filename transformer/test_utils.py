import numpy as np

# abs(a - b) < abs(a) * rel_tol
def rel_cmp(a, b, rel_tol=1e-5):
    if (np.shape(a) == np.shape(b)):
        return np.all(np.abs(a - b) <= np.abs(a) * rel_tol)
    else:
        print("shape inconsistent.")
        return False

from contextlib import contextmanager
@contextmanager
def printoptions(*args, **kwargs):
    original_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original_options)
