import numpy as np
from offline_rl_pusht.utils.normalizer import Standardizer


def test_standardizer_round_trip():
    x = np.array([[1.0, 3.0], [2.0, 5.0], [4.0, 9.0]], dtype=np.float32)
    norm = Standardizer.fit(x)
    recovered = norm.inverse_np(norm.transform_np(x))
    np.testing.assert_allclose(recovered, x, atol=1e-5)
