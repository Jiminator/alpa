# alpa_run.py

import scipy
try:
    from scipy.linalg import _basic
    scipy.linalg.tril = _basic.tril
    scipy.linalg.triu = _basic.triu
except Exception as e:
    print(f"[WARN] Could not monkey-patch tril/triu: {e}")

# Now safe to import jax and alpa
import alpa
import jax

# Optional: run test
import alpa.test_install