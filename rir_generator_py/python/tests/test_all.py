import numpy as np
import rir_generator_py

def test_nothing():
    imp = rir_generator_py.compute_rir(
        c=340.0,
        fs=16000.0,
        receiver=[2.0, 1.5, 2.0],
        microphone='o',
        angle=[0.0, 0.0],
        source=[2.0, 3.5, 2.0],
        room=[5.0, 4.0, 6.0],
        beta=[0.4] * 6,
        n_samples=4096,
        n_order=-1,
        enable_highpass_filter=True,
    );

    assert np.any(imp > 0.0)
