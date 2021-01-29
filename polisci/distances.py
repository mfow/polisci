import numpy as np


def get_distances(a: np.ndarray, b: np.ndarray, dist_type: str = 'l2'):
    assert len(a.shape) == 2
    assert len(b.shape) == 2
    assert a.shape[1] == b.shape[1]

    a = np.expand_dims(a, axis=1)
    b = np.expand_dims(b, axis=0)

    diffs = a - b

    if dist_type == 'l2':
        diffs = diffs * diffs

    if dist_type in ['l1', 'l2']:
        diffs = np.sum(diffs, axis=2)
    elif dist_type == 'inf':
        diffs = np.max(diffs, axis=2)
    else:
        raise RuntimeError('Unknown dist_type: {}'.format(dist_type))

    if dist_type == 'l2':
        diffs = np.sqrt(diffs)

    return diffs

