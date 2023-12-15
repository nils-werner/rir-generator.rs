import pytest
import numpy as np
import rir_generator_py
import rir_generator


@pytest.fixture(params=[True, False])
def hp_filter(request):
    return request.param


def test_nothing(hp_filter):
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
        enable_highpass_filter=hp_filter,
    );

    reference = rir_generator.generate(
        c=340.0,
        fs=16000.0,
        r=[2.0, 1.5, 2.0],
        mtype=rir_generator.mtype.omnidirectional,
        s=[2.0, 3.5, 2.0],
        L=[5.0, 4.0, 6.0],
        beta=[0.4] * 6,
        nsample=4096,
        order=-1,
        hp_filter=hp_filter,
    )

    assert np.allclose(imp.shape, reference.shape)
    assert np.allclose(imp, reference)
