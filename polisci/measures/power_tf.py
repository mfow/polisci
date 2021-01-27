import numpy as np
import tensorflow as tf
from ..models import load_polisci_model


__power_model_10 = load_polisci_model('political_power_2021-01-27-09-33-37.h5')
__model_parties = 10


def _get_sort_order(x: np.ndarray) -> np.ndarray:
    y = list(x)
    y.sort()

    result = list()

    def add_item(item):
        for i in range(len(x)):
            if x[i] == item and i not in result:
                result.append(i)
                return
        raise RuntimeError('Item not found')

    for item in y:
        add_item(item)

    return np.array(result, dtype=np.int)


def get_political_power_tensorflow(seats: np.ndarray, **kwargs) -> np.ndarray:
    if len(seats.shape) == 1:
        return get_political_power_tensorflow(np.expand_dims(seats, axis=0), **kwargs)

    assert len(seats.shape) == 2
    num_elections = seats.shape[0]
    num_parties = seats.shape[1]

    assert 2 <= num_parties <= __model_parties

    model_x = np.zeros([num_elections, __model_parties], dtype=np.float)
    sort_orders = np.zeros([num_elections, num_parties], dtype=np.int)
    num_dummy_parties = __model_parties - num_parties

    for election_id in range(num_elections):
        election_seats = seats[election_id, :]
        sort_order = _get_sort_order(election_seats)
        sort_orders[election_id, :] = sort_order
        election_seats = election_seats / (1.0 * np.sum(election_seats))
        model_x[election_id, num_dummy_parties:__model_parties] = election_seats[sort_order]

    model_y = __power_model_10(model_x)
    del model_x
    results = np.zeros([num_elections, num_parties], dtype=np.float)

    for election_id in range(num_elections):
        election_result_unsorted = model_y[election_id, num_dummy_parties:__model_parties]
        election_result_sorted = np.zeros([num_parties], dtype=np.float)
        sort_order = sort_orders[election_id, :]
        # todo: unshuffle election_result

        for i in range(num_parties):
            election_result_sorted[sort_order[i]] = election_result_unsorted[i]

        # we shouldn't have to normalize to 1 but do so anyway.
        election_result_sorted = election_result_sorted / np.sum(election_result_sorted)

        results[election_id, :] = election_result_sorted

    return results



